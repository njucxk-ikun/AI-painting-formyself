"""AI drawing workbench: grounded vocabulary, versioned prompts and image iteration."""
import hashlib
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from api_clients import DeepSeekClient, OpenAIImageClient, ImageGenerationError
from knowledge_base import KB_PATH, KnowledgeBase
from models import Project, ReferenceImage, SceneImagery
from storage import ROOT, WORKSPACE, atomic_json
from workflow import (attach_iteration, current_iteration, fork_iteration,
                      load_projects, new_project_id, project_path,
                       record_outputs, save_project, direction_content, develop_direction, prepare_generation,
                       initialize_direction_module_setup, module_term_candidates, plan_direction_imagery,
                       infer_detail_mode, imagery_validation_errors)
from prompt_catalog import (CONTENT_MODULE_IDS, MAX_CONTENT_MODULES, MIN_CONTENT_MODULES,
                            DETAIL_MODE_LABELS, MODULE_BY_ID, normalize_module_ids,
                            valid_content_selection)

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="AI绘画工作台", page_icon="🎨", layout="wide")


@st.cache_resource
def load_knowledge(stamp):
    return KnowledgeBase.load()


@st.cache_resource
def init_clients(stamp, model, effort, base_url, image_model, image_url):
    return (
        DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=base_url, model=model,
            reasoning_effort=effort, knowledge=load_knowledge(stamp),
            use_env_proxy=os.getenv("DEEPSEEK_USE_ENV_PROXY", "false").lower() == "true"),
        OpenAIImageClient(api_key=os.getenv("OPENAI_API_KEY"), base_url=image_url, model=image_model,
            quality=os.getenv("OPENAI_IMAGE_QUALITY", "high"),
            response_format=os.getenv("OPENAI_IMAGE_RESPONSE_FORMAT", "url"),
            use_env_proxy=os.getenv("OPENAI_USE_ENV_PROXY", "false").lower() == "true",
            output_dir=str(WORKSPACE / "temp_outputs")),
    )


def key(project, label, iteration=None):
    return f"{project.project_id}:{iteration.revision if iteration else 'project'}:{label}"


def switch_project(project):
    st.session_state.current_project = project
    st.session_state.page = "project"
    st.session_state.pop("feedback", None)
    st.session_state.pop("rewrite", None)
    st.session_state.pop("edit_direction_for", None)


def mark_changed(project):
    st.session_state.pop("feedback", None)
    st.session_state.pop("rewrite", None)
    iteration = current_iteration(project)
    if iteration and not (iteration.outputs or iteration.model_name or iteration.api_calls):
        iteration.needs_integration = True


def main():
    st.title("🎨 AI绘画工作台")
    st.caption("① 说想法 → ② 选方案与模块 → ③ 确认画面意象 → ④ 编辑最终提示词并出图 ↺ 继续修改、再次出图")
    st.session_state.setdefault("current_project", None)
    st.session_state.setdefault("page", "new")
    if "pending_project_id" in st.session_state:
        st.session_state.project_selector = st.session_state.pop("pending_project_id")
    try:
        stamp = KB_PATH.stat().st_mtime_ns
        kb = load_knowledge(stamp)
        deepseek, images = init_clients(stamp, os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            os.getenv("DEEPSEEK_REASONING_EFFORT", "medium"),
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
            os.getenv("OPENAI_BASE_URL", "https://xcode.best/v1"))
    except Exception as error:
        st.error(str(error))
        st.info("知识库首次构建：python build_knowledge.py")
        return
    projects, invalid = load_projects()
    if invalid:
        st.warning("无法读取部分项目，已保留原文件：" + "、".join(invalid))
    current = st.session_state.current_project
    with st.sidebar:
        st.header("项目管理")
        if st.button("➕ 新建项目", use_container_width=True):
            st.session_state.page = "new"
        options = [None] + [p.project_id for p in projects]
        labels = {p.project_id: p.display_name + " · " + p.project_id[:15] for p in projects}
        if "project_selector" not in st.session_state:
            st.session_state.project_selector = current.project_id if current and current.project_id in options else None
        selected = st.selectbox("选择项目", options=options, index=None,
            format_func=lambda value: labels.get(value, "请选择项目"), key="project_selector")
        if selected and (current is None or selected != current.project_id):
            switch_project(next(p for p in projects if p.project_id == selected))
            st.rerun()
        if st.button("查看术语库", use_container_width=True):
            st.session_state.page = "knowledge"
        if current and st.button("返回当前项目", use_container_width=True):
            st.session_state.page = "project"
        st.caption(f"知识库 {len(kb.terms):,} 条 · {len(kb.metadata['categories'])} 类")
        st.caption(f"绘画交互：{deepseek.model} · 思考 {deepseek.reasoning_effort}")
        if st.button("查询 DeepSeek 余额"):
            st.json(deepseek.get_balance())
        if st.button("查看图像服务配置"):
            st.json(images.get_usage_info())
    if st.session_state.page == "new":
        show_new_project(deepseek)
    elif st.session_state.page == "knowledge":
        show_knowledge_base(kb)
    elif st.session_state.current_project:
        show_project(st.session_state.current_project, deepseek, images, kb)
    else:
        st.info("从左侧新建项目，或先查看术语库。已有项目可从「选择项目」继续。")
        st.write("提交想法 → 选择并编辑方案 → 确认画面意象 → DeepSeek 深化 → 编辑提示词、提交出图 → 继续修改")


def show_new_project(deepseek):
    st.header("第 1 步 · 告诉我想画什么")
    st.write("先说想法，DeepSeek 会展开几个方案；这一步不会生成图片。")
    with st.form("new_project"):
        name = st.text_input("项目名称（可选）", placeholder="留空时根据想法命名")
        intent = st.text_area("创作意图", placeholder="描述主体、环境、动作或想表达的感觉。", height=150)
        mode = st.radio("模式", ["文生图", "图生图"], horizontal=True)
        submit = st.form_submit_button("提交想法，展开方案", type="primary")
    if submit:
        if not intent.strip():
            st.error("请先填写想画什么")
            return
        try:
            with st.spinner("DeepSeek 正在分析意图与相关视觉术语…"):
                mode_id = "img2img" if mode == "图生图" else "text2img"
                analysis = deepseek.analyze_intent(intent, mode_id)
                name = name.strip() or intent.strip()[:16]
                project = Project(project_id=new_project_id(name), short_name=name[:10],
                    display_name=name.strip(), initial_intent=intent.strip(), mode=mode_id, intent_analysis=analysis)
                save_project(project)
                switch_project(project)
                st.session_state.pending_project_id = project.project_id
                st.rerun()
        except Exception as error:
            st.error(str(error))


def show_project(project, deepseek, images, kb):
    st.header(project.display_name)
    iteration = current_iteration(project)
    editing_direction = st.session_state.get("edit_direction_for") == project.project_id
    if iteration is None or not iteration.locked_prompt or editing_direction:
        show_direction_step(project, deepseek, kb)
    else:
        show_prompt_step(project, deepseek, images)
    if project.iterations:
        with st.expander("历史提示词与图片（自动保存）"):
            for old in reversed(project.iterations):
                attempted = bool(old.outputs or old.model_name or old.api_calls or old.error_message)
                st.write(f"{old.revision} · {'已提交出图' if attempted else '提示词草稿'} · {len(old.outputs)} 张")
                if old.error_message:
                    st.caption(old.error_message)
                if old.direction_text:
                    st.caption("当时提交的方案：" + old.direction_text)
                if old.locked_prompt:
                    st.code(old.locked_prompt, language=None)
                for output in old.outputs:
                    if Path(output.image_path).exists():
                        st.image(output.image_path, width=180, caption=old.revision + " " + output.slot)
        with st.expander("成品与案例（可选）", expanded=bool(project.final_images)):
            show_finals(project)


def show_direction_step(project, deepseek, kb):
    st.subheader("第 2 步 · 选择并修改一个方案")
    st.write("选中方案后可修改正文、模块和知识词条。先让 DeepSeek 列出画面意象，逐项确认后再深化最终提示词。")
    analysis = project.intent_analysis
    if analysis is None:
        st.info(project.initial_intent)
        if st.button("让 DeepSeek 展开方案", key=key(project, "analyze"), type="primary"):
            try:
                with st.spinner("DeepSeek 正在展开方案…"):
                    project.intent_analysis = deepseek.analyze_intent(
                        project.initial_intent, project.mode, project.selected_term_ids)
                save_project(project)
                st.rerun()
            except Exception as error:
                st.error(str(error))
        return
    st.caption("最初想法：" + project.initial_intent)
    st.write(analysis.restated_goal)
    with st.expander("比较 DeepSeek 展开的全部方案", expanded=not bool(project.selected_direction)):
        for index, direction in enumerate(analysis.visual_directions, 1):
            st.markdown(f"**方案 {index} · {direction.name}{'（推荐）' if direction.is_default else ''}**")
            st.write(direction_content(direction))
            st.caption("选择理由：" + direction.applicable_reason)
            if direction.technical_terms:
                st.caption("相关视觉术语：" + "、".join(direction.technical_terms))
    if analysis.main_challenges:
        with st.expander("DeepSeek 对这次创作的理解与取舍"):
            for challenge in analysis.main_challenges:
                st.write("· " + challenge)
    directions = {d.direction_id: d for d in analysis.visual_directions}
    ids = list(directions)
    selected = st.radio("选择一个展开方案", ids,
        index=ids.index(project.selected_direction) if project.selected_direction in ids else None,
        format_func=lambda identity: directions[identity].name, key=key(project, "direction-choice"))
    submitted_text = None
    modules_valid = False
    imagery_current = False
    if selected:
        submitted_text = st.text_area("方案内容（可修改）",
            value=project.direction_drafts.get(selected, direction_content(directions[selected])),
            height=260, key=key(project, "direction-text:" + selected))
        if project.direction_drafts.get(selected) != submitted_text:
            project.direction_drafts[selected] = submitted_text
            save_project(project)
        modules_valid = show_prompt_module_setup(project, selected, directions[selected], submitted_text, kb)
        current_mode = infer_detail_mode(project, selected, submitted_text, kb)
        mode_ids = list(DETAIL_MODE_LABELS)
        chosen_mode = st.radio("画面细节密度", mode_ids, horizontal=True,
            index=mode_ids.index(current_mode), format_func=lambda value: DETAIL_MODE_LABELS[value],
            key=key(project, "detail-mode:" + selected),
            help="二次元和轻小说默认使用干净克制；只有明确需要复杂环境与材质时再选择丰富细节。")
        if chosen_mode != current_mode:
            project.direction_detail_modes[selected] = chosen_mode
            save_project(project)
            current_mode = chosen_mode
        st.caption({
            "clean": "主体和角色优先，背景简化，限制粒子、泛光、复杂纹理与电影化效果。",
            "balanced": "保留必要环境和光色层次，但背景与装饰不能抢过主体。",
            "rich": "允许增加与主题直接相关的材质和环境层次，仍需保持清楚主次。",
        }[current_mode])
        imagery_current = bool(project.direction_imagery.get(selected)) \
            and project.direction_imagery_source_text.get(selected) == submitted_text.strip() \
            and project.direction_imagery_detail_modes.get(selected) == current_mode
    else:
        st.info("先在上方选中一个方案，再查看并修改正文。")
    show_constraints(project)
    if project.mode == "img2img" or project.reference_images:
        show_references(project)
    names = [kb.by_id[t]["name"] for t in project.selected_term_ids if t in kb.by_id]
    if names:
        st.caption("项目候选术语（会在上方按模块分配，可继续调整）：" + "、".join(names))
    missing_refs = project.mode == "img2img" and not project.reference_images
    if missing_refs:
        st.info("图生图请先在「参考图片与用途」上传参考图。")
    if selected and project.direction_imagery.get(selected) and not imagery_current:
        st.warning("方案正文或细节密度已经变化，需要重新生成意象清单后再深化。")
    if st.button("提交方案，让 DeepSeek 生成意象清单", type="primary",
                 key=key(project, "plan-imagery"), disabled=not selected or not modules_valid or missing_refs):
        try:
            with st.spinner("DeepSeek 正在列出画面中所有需要确认的意象…"):
                plan_direction_imagery(project, deepseek, kb, selected, submitted_text)
            save_project(project)
            st.rerun()
        except Exception as error:
            st.error(str(error))
            st.info("已保留方案编辑内容和原有提示词，可直接重试。")
    imagery_errors = []
    if selected and imagery_current:
        imagery_errors = show_imagery_step(project, selected)
        if st.button("确认保留的意象，让 DeepSeek 深化提示词", type="primary",
                     key=key(project, "develop"), disabled=bool(imagery_errors)):
            try:
                with st.spinner("DeepSeek 正在依据已确认意象整理分区提示词…"):
                    develop_direction(project, deepseek, kb, selected, submitted_text)
                save_project(project)
                st.session_state.pop("edit_direction_for", None)
                st.session_state.pop("feedback", None)
                st.rerun()
            except Exception as error:
                st.error(str(error))
                st.info("已保留意象选择、方案编辑内容和原有提示词，可直接重试。")
    if current_iteration(project) and current_iteration(project).locked_prompt:
        if st.button("返回已有提示词，继续编辑", key=key(project, "cancel-direction")):
            st.session_state.pop("edit_direction_for", None)
            st.rerun()


def show_imagery_step(project, direction_id):
    st.subheader("第 3 步 · 选择并明确画面意象")
    st.write("取消不需要的意象；保留项必须写清自然数量或结构、位置、外观、状态或动作，以及与其他元素的关系。")
    st.caption("复杂建筑、连续结构和群体不要机械写成‘一个’；最终提示词会把这些元素组织成完整场景，而不是逐项拼贴。")
    imagery = project.direction_imagery[direction_id]
    categories = ["subject", "character", "environment", "object", "action", "visual_symbol", "effect"]
    category_labels = {
        "subject": "主体", "character": "人物", "environment": "环境", "object": "物件",
        "action": "动作", "visual_symbol": "视觉符号", "effect": "画面效果",
    }
    changed = False
    for item in imagery:
        marker = "备选 · " if item.optional_candidate else "核心 · "
        selected = st.checkbox(marker + item.name, value=item.selected,
            key=key(project, "imagery-selected:" + direction_id + ":" + item.imagery_id))
        if selected != item.selected:
            item.selected = selected
            changed = True
        with st.expander(item.name + (" · 已保留" if item.selected else " · 已取消"), expanded=item.selected):
            category = st.selectbox("类别", categories, index=categories.index(item.category),
                format_func=lambda value: category_labels[value],
                key=key(project, "imagery-category:" + direction_id + ":" + item.imagery_id))
            name = st.text_input("意象名称", value=item.name,
                key=key(project, "imagery-name:" + direction_id + ":" + item.imagery_id))
            quantity_modes = ["exact", "aggregate", "continuous", "distributed"]
            quantity_labels = {"exact": "明确数量", "aggregate": "整体/聚合结构", "continuous": "连续结构", "distributed": "分布式群体"}
            quantity_mode = st.selectbox("数量/结构类型", quantity_modes,
                index=quantity_modes.index(item.quantity_mode), format_func=lambda value: quantity_labels[value],
                key=key(project, "imagery-quantity-mode:" + direction_id + ":" + item.imagery_id))
            quantity_default = item.quantity_description or (f"{item.exact_count}个独立对象" if item.exact_count else "")
            quantity = st.text_input("自然数量或结构描述", value=quantity_default,
                key=key(project, "imagery-quantity:" + direction_id + ":" + item.imagery_id),
                placeholder="例如：一整栋连续的老旧公寓楼；成片密集排列的窗户")
            position = st.text_input("画面位置", value=item.position,
                key=key(project, "imagery-position:" + direction_id + ":" + item.imagery_id))
            appearance = st.text_area("可见外观", value=item.appearance, height=90,
                key=key(project, "imagery-appearance:" + direction_id + ":" + item.imagery_id))
            state_action = st.text_input("状态或动作", value=item.state_action,
                key=key(project, "imagery-state:" + direction_id + ":" + item.imagery_id))
            relationship = st.text_input("与其他意象的关系", value=item.relationship,
                key=key(project, "imagery-relation:" + direction_id + ":" + item.imagery_id))
            values = (category, name, quantity_mode, quantity, position, appearance, state_action, relationship)
            old = (item.category, item.name, item.quantity_mode, item.quantity_description, item.position,
                   item.appearance, item.state_action, item.relationship)
            if values != old:
                (item.category, item.name, item.quantity_mode, item.quantity_description, item.position,
                 item.appearance, item.state_action, item.relationship) = values
                changed = True
            if item.reason:
                st.caption("DeepSeek 选择理由：" + item.reason)
    if st.button("添加自定义意象", key=key(project, "add-imagery:" + direction_id)):
        next_index = max([int(item.imagery_id.rsplit("_", 1)[-1]) for item in imagery
                          if item.imagery_id.rsplit("_", 1)[-1].isdigit()] or [0]) + 1
        imagery.append(SceneImagery(imagery_id=f"img_{next_index:03d}", category="object", name="新意象",
            exact_count=None, quantity_description="请填写自然数量或结构", quantity_mode="aggregate",
            position="请填写明确位置", appearance="请填写可见外观",
            state_action="请填写状态或动作", relationship="请填写与其他意象的关系",
            importance="supporting", reason="用户自定义"))
        save_project(project)
        st.rerun()
    if changed:
        save_project(project)
    errors = imagery_validation_errors(imagery)
    for error in errors:
        st.error(error)
    return errors


def show_prompt_module_setup(project, direction_id, direction, submitted_text, kb):
    before = (
        project.direction_module_ids.get(direction_id),
        project.direction_module_term_ids.get(direction_id),
        project.direction_module_notes.get(direction_id),
        project.direction_constraint_modules.get(direction_id),
    )
    initialize_direction_module_setup(project, direction_id, submitted_text, kb)
    after = (
        project.direction_module_ids.get(direction_id),
        project.direction_module_term_ids.get(direction_id),
        project.direction_module_notes.get(direction_id),
        project.direction_constraint_modules.get(direction_id),
    )
    if before != after:
        save_project(project)

    st.markdown("#### 提示词模块与知识库素材")
    st.caption("DeepSeek 已从固定目录推荐内容模块。可以增删到 6–8 个；每个模块最多选择 3 个知识词条，所见内容就是下一步实际上下文。")
    selected_ids = project.direction_module_ids[direction_id]
    chosen = st.multiselect(
        "内容模块（选择 6–8 个）",
        options=list(CONTENT_MODULE_IDS),
        default=selected_ids,
        format_func=lambda identity: MODULE_BY_ID[identity].title,
        key=key(project, "module-selection:" + direction_id),
    )
    normalized = normalize_module_ids(chosen, content_only=True)
    if normalized != selected_ids:
        project.direction_module_ids[direction_id] = normalized
        term_map = project.direction_module_term_ids.setdefault(direction_id, {})
        note_map = project.direction_module_notes.setdefault(direction_id, {})
        for identity in normalized:
            if identity not in term_map:
                candidates = module_term_candidates(kb, identity,
                    submitted_text + " " + " ".join(direction.technical_terms), limit=1)
                term_map[identity] = [term["id"] for term in candidates[:1]]
            note_map.setdefault(identity, "")
        save_project(project)
        selected_ids = normalized
    if not valid_content_selection(selected_ids):
        st.error(f"当前选择 {len(selected_ids)} 个内容模块；提交前需要保留 {MIN_CONTENT_MODULES}–{MAX_CONTENT_MODULES} 个，并包含“场景与细节”。")

    constraint_ids = project.direction_constraint_modules.setdefault(direction_id, [])
    keep_separate = st.checkbox("单列「绝对需要遵守的内容」",
        value="must_keep" in constraint_ids, key=key(project, "constraint-keep:" + direction_id))
    avoid_separate = st.checkbox("单列「绝对需要避免的内容」",
        value="must_avoid" in constraint_ids, key=key(project, "constraint-avoid:" + direction_id),
        help="不勾选时，已填写的避免项会合并到「绝对需要遵守的内容」；没有硬约束时两项都可不选。")
    new_constraints = (["must_keep"] if keep_separate else []) + (["must_avoid"] if avoid_separate else [])
    if new_constraints != constraint_ids:
        project.direction_constraint_modules[direction_id] = new_constraints
        for identity in new_constraints:
            project.direction_module_notes.setdefault(direction_id, {}).setdefault(identity, "")
        save_project(project)
        constraint_ids = new_constraints

    term_map = project.direction_module_term_ids.setdefault(direction_id, {})
    note_map = project.direction_module_notes.setdefault(direction_id, {})
    style_candidate_map = project.direction_style_candidate_ids.setdefault(direction_id, [])
    for module_id in [*selected_ids, *constraint_ids]:
        spec = MODULE_BY_ID[module_id]
        current_ids = term_map.setdefault(module_id, [])
        with st.expander(spec.title + (f" · {len(current_ids)} 个知识词条" if current_ids else "")):
            st.caption(spec.purpose)
            if module_id == "core_style" and direction.style_candidates:
                st.markdown("**可选风格参考（默认不添加）**")
                st.caption("这些是风格方向候选，不会自动进入提示词；可选择多个，但请确认它们能够兼容。")
                candidate_ids = [candidate.candidate_id for candidate in direction.style_candidates]
                chosen_candidates = st.multiselect(
                    "核心风格候选",
                    options=candidate_ids,
                    default=[identity for identity in style_candidate_map if identity in candidate_ids],
                    format_func=lambda identity: next(candidate.name for candidate in direction.style_candidates
                                                      if candidate.candidate_id == identity),
                    max_selections=6,
                    key=key(project, "style-candidates:" + direction_id),
                )
                if chosen_candidates != style_candidate_map:
                    project.direction_style_candidate_ids[direction_id] = chosen_candidates
                    style_candidate_map = chosen_candidates
                    save_project(project)
                for candidate in direction.style_candidates:
                    if candidate.candidate_id in style_candidate_map:
                        english = f" / {candidate.english_name}" if candidate.english_name else ""
                        st.caption(f"已选：{candidate.name}{english} · {candidate.rationale}")
            if spec.knowledge_categories:
                query = st.text_input(spec.title + "术语搜索", key=key(project, "module-search:" + direction_id + ":" + module_id),
                    placeholder="输入名称、英文或画面特征；留空时按当前方案推荐")
                search_text = query.strip() or (submitted_text + " " + " ".join(direction.technical_terms))
                candidates = module_term_candidates(kb, module_id, search_text, current_ids, limit=12)
                option_ids = [term["id"] for term in candidates]
                chosen_terms = st.multiselect(spec.title + "知识库术语", options=option_ids,
                    default=[identity for identity in current_ids if identity in option_ids],
                    format_func=lambda identity: kb.by_id[identity]["name"], max_selections=3,
                    key=key(project, "module-terms:" + direction_id + ":" + module_id))
                if chosen_terms != current_ids:
                    term_map[module_id] = chosen_terms
                    save_project(project)
                    current_ids = chosen_terms
                for term_id in current_ids:
                    term = kb.by_id.get(term_id)
                    if term:
                        st.markdown("**" + term["name"] + "** · " + term["visual_effect"])
                        st.caption("可直接使用：" + "、".join(term["prompt_keywords"]) + "｜识别要点：" + "；".join(term["visual_cues"][:3]))
            note = st.text_area(spec.title + "补充要求（可选）", value=note_map.get(module_id, ""),
                key=key(project, "module-note:" + direction_id + ":" + module_id),
                placeholder="例如：保留大面积留白；主体必须在右侧。")
            if note != note_map.get(module_id, ""):
                note_map[module_id] = note
                save_project(project)
    return valid_content_selection(selected_ids)


def show_constraints(project):
    with st.expander("可选：提前写下必须保留和避免的内容"):
        keep = st.text_area("必须保留", project.must_keep, key=key(project, "must-keep"))
        avoid = st.text_area("必须避免", project.must_avoid, key=key(project, "must-avoid"))
        locked = st.text_area("禁止改动", project.locked_constraints, key=key(project, "locked"))
        if (keep, avoid, locked) != (project.must_keep, project.must_avoid, project.locked_constraints):
            project.must_keep, project.must_avoid, project.locked_constraints = keep, avoid, locked
            mark_changed(project)
            save_project(project)
        st.caption("随方案一起提交给 DeepSeek；下一步也可以直接修改最终提示词中的约束分区。")


def show_references(project):
    with st.expander("参考图片与用途", expanded=project.mode == "img2img"):
        st.caption("文字策划依据用途说明；实际绘画时，图片会发送到图像编辑接口。")
        with st.form(key(project, "upload"), clear_on_submit=True):
            uploaded = st.file_uploader("添加参考图", type=["png", "jpg", "jpeg", "webp"])
            labels = {"character": "角色", "composition": "构图", "style": "风格", "lighting": "光影", "color": "配色"}
            role = st.selectbox("参考用途", options=list(labels), format_func=lambda v: labels[v])
            description = st.text_input("需要参考什么", placeholder="例如：只保留角色服装与发型")
            submit = st.form_submit_button("保存参考图")
        if submit and uploaded:
            try:
                content = uploaded.getvalue()
                if len(content) > 20 * 1024 * 1024:
                    raise ValueError("参考图请小于20 MB")
                Image.open(BytesIO(content)).verify()
                digest = hashlib.sha256(content).hexdigest()
                if any(r.file_hash == digest and r.role.value == role for r in project.reference_images):
                    raise ValueError("已添加同一图片和用途")
                directory = project_path(project.project_id).parent / "references"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (digest[:20] + Path(uploaded.name).suffix.lower())
                path.write_bytes(content)
                project.reference_images.append(ReferenceImage(ref_id=digest[:12] + "_" + role,
                    file_path=str(path), file_hash=digest, role=role, description=description))
                mark_changed(project)
                save_project(project)
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
        for ref in project.reference_images:
            if Path(ref.file_path).exists():
                st.image(ref.file_path, width=140, caption=ref.role.value + " · " + ref.description)
            else:
                st.warning("参考图文件不存在：" + Path(ref.file_path).name)
            if st.button("移除参考", key=key(project, "remove:" + ref.ref_id)):
                project.reference_images.remove(ref)
                mark_changed(project)
                save_project(project)
                st.rerun()


def show_prompt_step(project, deepseek, images):
    st.subheader("第 4 步 · 修改最终提示词，提交出图")
    iteration = current_iteration(project)
    st.write("下方就是实际用于绘图的最终提示词。可以修改，也可以不改，直接点击「提交提示词，生成 A/B 图片」。")
    st.caption("修改会自动保存。出图后可以在这里继续改稿、再次提交；每轮提示词和图片自动保存为历史。")
    if st.button("返回方案，重新选择或修改", key=key(project, "back-to-direction")):
        st.session_state.edit_direction_for = project.project_id
        st.rerun()
    if iteration.prompt_modules:
        st.caption("DeepSeek 细化结构：" + " → ".join(m.title for m in iteration.prompt_modules))
    if iteration.contains_people and project.mode == "text2img":
        st.info("这是人物文生图。提交时会强制保留“禁止多余或错误肢体”的解剖约束。")
    prompt_key = key(project, "final-prompt:" + str(project.prompt_draft_version))
    initial = project.prompt_draft if project.prompt_draft is not None else iteration.locked_prompt
    prompt = st.text_area("最终提示词（可直接修改）", value=initial, height=520, key=prompt_key)
    if project.prompt_draft != prompt:
        project.prompt_draft = prompt
        st.session_state.pop("feedback", None)
        save_project(project)
    if iteration.needs_integration:
        st.info("这是旧版待整合的提示词。请在此检查、修改后直接提交；也可返回方案重新深化。")
    for diff in iteration.prompt_diff:
        st.warning(f"DeepSeek 提醒 · {diff.module}：{diff.reason}\n建议：{diff.new_content or ''}")
    ratios = ["1:1", "16:9", "9:16", "4:3", "3:4"]
    preferred = project.draft_aspect_ratio or iteration.aspect_ratio
    ratio = st.selectbox("画幅比例", ratios, index=ratios.index(preferred) if preferred in ratios else 0,
                         key=key(project, "draft-ratio"))
    if project.draft_aspect_ratio != ratio:
        project.draft_aspect_ratio = ratio
        save_project(project)
    size = images.size_for_aspect_ratio(ratio)
    st.caption(f"本次 A/B 各请求一次，可能产生两次计费 · {images.model} · {images.quality} · 请求尺寸 {size}，实际像素以返回图片为准。")
    missing_refs = project.mode == "img2img" and not iteration.reference_images
    if missing_refs:
        st.warning("图生图缺少参考图，请返回方案添加参考图后重新提交深化。")
    if st.button("提交提示词，生成 A/B 图片", key=key(project, "submit-images"), type="primary",
                 disabled=not prompt.strip() or missing_refs):
        try:
            target = prepare_generation(project, prompt, ratio, size)
            if target.locked_prompt != prompt:
                st.info("已自动恢复人物肢体约束，并同步写回最终提示词。")
            st.session_state.pop("feedback", None)
            save_project(project)
            with st.spinner("正在按当前提示词生成 A/B 图片，完成后可在原编辑区继续修改…"):
                generate_pair(project, target, images)
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if iteration.error_message:
        st.error(iteration.error_message)
        st.caption("编辑稿和已成功图片均已保留，检查后可直接再次提交。")
    show_latest_outputs(project)
    if any(it.outputs for it in project.iterations) and iteration.prompt_modules:
        show_feedback_assistant(project, iteration, deepseek, prompt)
    with st.expander("查看 DeepSeek 原始分区说明（供参考）"):
        st.caption("这是深化时的原始说明；实际绘图始终使用上方当前编辑稿。")
        for module in iteration.prompt_modules:
            st.markdown("**" + module.title + "**")
            st.write(module.confirmed_content or module.ai_draft)
            if module.recommend_reason:
                st.caption(module.recommend_reason)


def show_latest_outputs(project):
    iteration = next((it for it in reversed(project.iterations) if it.outputs), None)
    if iteration is None:
        return
    st.subheader("生成结果 · " + iteration.revision)
    if current_iteration(project).revision != iteration.revision:
        st.caption("以下为上一轮结果，供修改参考；当前编辑稿尚未出图。")
    else:
        st.info("图片已返回。可以继续修改上方提示词再提交，或将满意的图片设为成品。")
    cols = st.columns(2)
    for index, output in enumerate(iteration.outputs[-2:]):
        with cols[index]:
            path = Path(output.image_path)
            if not path.exists():
                st.warning("图片文件已丢失：" + path.name)
                continue
            dimensions = f" · {output.actual_width} × {output.actual_height}" if output.actual_width and output.actual_height else ""
            st.image(str(path), caption=output.slot + dimensions)
            st.download_button("下载 " + output.slot, data=path.read_bytes(), file_name=path.name, mime="image/png",
                               key=key(project, "download:" + output.slot, iteration))
            if st.button("设为成品 " + output.slot, key=key(project, "final:" + output.slot, iteration)):
                if output.image_path not in project.final_images:
                    project.final_images.append(output.image_path)
                output.selected_as_final = True
                save_project(project)
                st.rerun()


def show_feedback_assistant(project, iteration, deepseek, current_prompt):
    with st.expander("可选：让 DeepSeek 帮我修改提示词"):
        st.caption("直接手改上方提示词即可再次出图。也可以描述问题，让 DeepSeek 提出修改稿，再自行决定是否采用。")
        with st.form(key(project, "feedback-form", iteration)):
            problems = st.text_area("想让 DeepSeek 怎么改")
            keep = st.text_area("希望保留的点")
            submit = st.form_submit_button("让 DeepSeek 提出修改稿")
        if submit:
            try:
                if not problems.strip():
                    raise ValueError("请填写需要修改的问题")
                if not current_prompt.strip():
                    raise ValueError("当前提示词不能为空")
                with st.spinner("DeepSeek 正在根据当前编辑稿提出修改…"):
                    diagnosis, diffs, new_prompt = deepseek.analyze_feedback(
                        current_prompt, problems, keep, iteration.prompt_modules,
                        "以当前最终提示词中最新的约束为准，不得恢复用户已从最终稿删除或修改的旧约束。",
                        iteration.knowledge_term_ids or project.selected_term_ids,
                        planning_context=iteration.planning_snapshot)
                st.session_state.feedback = {"scope": key(project, "feedback", iteration),
                    "baseline": current_prompt, "diagnosis": diagnosis, "diffs": diffs,
                    "prompt": new_prompt, "problems": problems, "keep": keep}
                st.rerun()
            except Exception as error:
                st.error(str(error))
        result = st.session_state.get("feedback")
        if not result or result["scope"] != key(project, "feedback", iteration) or result["baseline"] != current_prompt:
            return
        st.write(result["diagnosis"])
        for diff in result["diffs"]:
            st.write(f"{diff.module}：{diff.reason}")
            st.code(f"原：{diff.old_content or ''}\n改：{diff.new_content or ''}", language=None)
        st.code(result["prompt"], language=None)
        if st.button("采用到提示词编辑区", key=key(project, "adopt-feedback", iteration)):
            try:
                target = fork_iteration(project, iteration, "feedback", prompt=result["prompt"], diffs=result["diffs"])
                target.user_feedback = target.change_requests = result["problems"]
                target.keep_aspects = result["keep"]
                target.interaction_model = deepseek.model
                attach_iteration(project, target)
                project.prompt_draft = result["prompt"]
                project.prompt_draft_version += 1
                save_project(project)
                st.session_state.pop("feedback", None)
                st.rerun()
            except Exception as error:
                st.error(str(error))
        if st.button("放弃这份修改稿", key=key(project, "reject-feedback", iteration)):
            st.session_state.pop("feedback", None)
            st.rerun()


def generate_pair(project, iteration, images):
    iteration.model_name = images.model
    save_project(project)
    try:
        paths = images.generate_images(iteration.locked_prompt, num_images=2, aspect_ratio=iteration.aspect_ratio,
            resolution=iteration.output_size,
            reference_images=[(r.file_path, r.role.value + "：" + r.description) for r in iteration.reference_images])
        record_outputs(iteration, paths, 2)
    except ImageGenerationError as error:
        record_outputs(iteration, error.saved_paths, error.api_calls)
        iteration.error_message = str(error)
        iteration.success = False
    except Exception as error:
        iteration.error_message = str(error)
        iteration.success = False
    save_project(project)





def show_finals(project):
    st.subheader("成品归档")
    for index, image_path in enumerate(list(project.final_images)):
        if Path(image_path).exists():
            st.image(image_path, width=480)
        else:
            st.warning("成品文件不存在：" + Path(image_path).name)
        if st.button("移除此成品", key=key(project, "remove-final:" + str(index))):
            project.final_images.remove(image_path)
            for iteration in project.iterations:
                for output in iteration.outputs:
                    if output.image_path == image_path:
                        output.selected_as_final = False
            save_project(project)
            st.rerun()
    if not project.final_images:
        st.info("在生成结果中选择满意的图片设为成品。")
        return
    if st.button("保存为案例", key=key(project, "case")):
        matched = [it for it in project.iterations if any(o.image_path in project.final_images for o in it.outputs)]
        for item in matched:
            atomic_json(WORKSPACE / "library" / (project.project_id + "_" + item.revision + ".json"), {
                "project_id": project.project_id, "revision": item.revision, "title": project.display_name,
                "intent": project.initial_intent, "prompt": item.locked_prompt,
                "modules": [m.model_dump(mode="json") for m in item.prompt_modules],
                "knowledge_term_ids": item.knowledge_term_ids,
                "final_images": [o.image_path for o in item.outputs if o.image_path in project.final_images],
                "saved_at": datetime.now().isoformat()})
        st.success(f"已保存 {len(matched)} 个版本的案例快照")


def show_knowledge_base(kb):
    st.header("视觉语言知识库")
    st.caption("解释与示例由 DeepSeek 生成；已做结构校验，尚未逐条人工审校。")
    query = st.text_input("搜索术语、英文或画面特征", key="kb-query", placeholder="例如：湿润街道、几何海报、柔和阴影")
    categories = {c["id"]: c["name"] for c in kb.metadata["categories"]}
    category = st.selectbox("分类", [None] + list(categories), key="kb-category",
                            format_func=lambda c: categories.get(c, "全部"))
    matches = kb.search(query, limit=len(kb.terms), category=category)
    signature = (query, category)
    if st.session_state.get("kb-filter") != signature:
        st.session_state["kb-page"] = 1
        st.session_state["kb-filter"] = signature
    pages = max(1, (len(matches) + 19) // 20)
    page = st.number_input("页码", min_value=1, max_value=pages, step=1, key="kb-page")
    st.caption(f"找到 {len(matches)} 条 · 第 {page}/{pages} 页 · 每页20条")
    project = st.session_state.current_project
    if project:
        selected = [kb.by_id[i]["name"] for i in project.selected_term_ids if i in kb.by_id]
        st.info("当前项目「" + project.display_name + "」已选：" + ("、".join(selected) or "无"))
    for term in matches[(page - 1) * 20:page * 20]:
        with st.expander(term["name"] + " · " + term["english_name"] + " · " + term["category_name"]):
            st.caption(term["nature"] + "｜别名：" + "、".join(term["aliases"]))
            st.write(term["description"])
            st.write("视觉效果：" + term["visual_effect"])
            st.write("识别要点：" + "；".join(term["visual_cues"]))
            st.write("适用：" + "、".join(term["applicable_scenes"]))
            st.write("不适用：" + "；".join(term["avoid_when"]))
            st.code(", ".join(term["prompt_keywords"]), language=None)
            st.write("绘画示例：" + term["examples"][0])
            for relation in term["confused_with"]:
                st.write("易混淆 · " + relation["term"] + "：" + relation["note"])
            for relation in term["combinations"]:
                st.write("组合 · " + relation["term"] + "：" + relation["note"])
            st.warning("；".join(term["cautions"]))
            p = term["provenance"]
            provenance_parts = [term["id"], p.get("model", "")]
            if p.get("generated_at"):
                provenance_parts.append(p["generated_at"][:10])
            st.caption(" · ".join(part for part in provenance_parts if part))
            if p.get("ai_review"):
                st.caption("已进行 DeepSeek 技术复核；尚未经专家审校。")
            if project:
                included = term["id"] in project.selected_term_ids
                if st.button("从项目移除" if included else "加入当前项目",
                             key="kb-add:" + project.project_id + ":" + term["id"]):
                    if included:
                        project.selected_term_ids.remove(term["id"])
                    elif len(project.selected_term_ids) < 10:
                        project.selected_term_ids.append(term["id"])
                    else:
                        st.error("每个项目最多选择10个重点术语，避免互相矛盾或冲淡主体。")
                        continue
                    mark_changed(project)
                    save_project(project)
                    st.rerun()


if __name__ == "__main__":
    main()
