"""Persistence and iteration transitions shared by the UI and regression tests."""
import re
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image

from models import DynamicModule, GenerationOutput, Iteration, Project, SceneImagery
from storage import WORKSPACE, atomic_json
from knowledge_base import SCHEMA_VERSION
from prompt_catalog import (ANATOMY_CONSTRAINT, CONSTRAINT_MODULE_IDS, MAX_CONTENT_MODULES,
                            MIN_CONTENT_MODULES, MODULE_BY_ID, detail_mode_policy,
                            recommended_content_modules,
                            normalize_module_ids, valid_content_selection)

PROJECTS = WORKSPACE / "projects"
PEOPLE_PATTERN = re.compile(
    r"人物|人像|女孩|男孩|少女|少年|女人|男人|女性|男性|旅人|行人|学生|老师|母亲|父亲|"
    r"儿童|老人|人群|人类|person|people|woman|women|man|men|girl|boy|human",
    re.IGNORECASE,
)
NO_PEOPLE_PATTERN = re.compile(r"无人物|没有人物|不出现人物|空无一人|no people|no person|without people", re.IGNORECASE)
VAGUE_IMAGERY_PATTERN = re.compile(r"若干|一些|可能|大概|之类|等等|任选|任意|或许|或者")


def new_project_id(name):
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:20] or "project"
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_" + safe


def project_path(project_id):
    target = (PROJECTS / project_id / "project.json").resolve()
    if PROJECTS.resolve() not in target.parents:
        raise ValueError("无效的项目路径")
    return target


def save_project(project):
    project.updated_at = datetime.now()
    atomic_json(project_path(project.project_id), project.model_dump(mode="json"))


def load_projects():
    projects, errors = [], []
    if PROJECTS.exists():
        for path in PROJECTS.glob("*/project.json"):
            try:
                project = Project.model_validate_json(path.read_text(encoding="utf-8"))
                if project.project_id != path.parent.name:
                    raise ValueError("项目ID与目录不匹配")
                projects.append(project)
            except (OSError, ValueError):
                errors.append(path.parent.name)
    return sorted(projects, key=lambda p: p.updated_at, reverse=True), errors


def current_iteration(project):
    return next((it for it in project.iterations if it.revision == project.current_revision),
                project.iterations[-1] if project.iterations else None)


def constraints_for(project):
    return f"必须保留：{project.must_keep}\n必须避免：{project.must_avoid}\n锁定不可改动：{project.locked_constraints}"


def next_revision(project):
    numbers = [int(it.revision[1:]) for it in project.iterations if re.fullmatch(r"r\d+", it.revision)]
    return f"r{max(numbers, default=0) + 1:03d}"


def fork_iteration(project, source, iteration_type, prompt=None, diffs=None):
    target = source.model_copy(deep=True)
    target.revision = next_revision(project)
    target.iteration_type = iteration_type
    target.created_at = datetime.now()
    target.outputs = []
    target.api_calls = 0
    target.success = False
    target.error_message = ""
    target.locked_prompt = prompt if prompt is not None else source.locked_prompt
    target.prompt_diff = diffs or []
    target.needs_integration = source.needs_integration if prompt is None else False
    for diff in diffs or []:
        match = next((m for m in target.prompt_modules if m.title == diff.module), None)
        if not match or not diff.new_content:
            raise ValueError("反馈修改缺少对应模块或完整内容")
        match.confirmed_content = diff.new_content
        match.last_updated = datetime.now()
    return target


def attach_iteration(project, iteration):
    if any(it.revision == iteration.revision for it in project.iterations):
        raise ValueError("迭代版本号重复")
    project.iterations.append(iteration)
    project.current_revision = iteration.revision


def direction_content(direction):
    """Old projects lack expanded_idea; their existing explanation stays editable."""
    return direction.expanded_idea.strip() or f"{direction.name}\n\n{direction.explanation}\n\n画面效果：{direction.visual_consequences}"


def prompt_contains_people(text):
    cleaned = NO_PEOPLE_PATTERN.sub("", text or "")
    return bool(PEOPLE_PATTERN.search(cleaned))


def direction_contains_people(project, direction_id, submitted_text=""):
    direction = next((item for item in project.intent_analysis.visual_directions
                      if item.direction_id == direction_id), None) if project.intent_analysis else None
    imagery = project.direction_imagery.get(direction_id, [])
    return bool((direction and direction.contains_people) or
                prompt_contains_people(submitted_text) or
                prompt_contains_people(" ".join(
                    f"{item.name} {item.appearance} {item.state_action}" for item in imagery if item.selected)))


def infer_detail_mode(project, direction_id, submitted_text, kb=None):
    existing = project.direction_detail_modes.get(direction_id)
    if existing:
        return existing
    direction = next((item for item in project.intent_analysis.visual_directions
                      if item.direction_id == direction_id), None) if project.intent_analysis else None
    text = " ".join([submitted_text, direction.name if direction else "",
                     " ".join(direction.technical_terms) if direction else ""]).lower()
    selected = project.direction_module_term_ids.get(direction_id, {})
    if kb:
        text += " " + " ".join(kb.by_id[term_id]["name"] for ids in selected.values()
                                  for term_id in ids if term_id in kb.by_id)
    mode = "clean" if any(word in text for word in (
        "二次元", "动漫", "动画", "轻小说", "anime", "manga", "visual novel", "日系插画")) else "balanced"
    project.direction_detail_modes[direction_id] = mode
    return mode


def imagery_validation_errors(imagery):
    errors = []
    selected = [item for item in imagery if item.selected]
    if not selected:
        return ["至少保留一个画面意象"]
    required = (("名称", "name"), ("数量或结构", "quantity_description"), ("位置", "position"), ("外观", "appearance"),
                ("状态或动作", "state_action"), ("关系", "relationship"))
    for item in selected:
        for label, field in required:
            value = getattr(item, field, "").strip()
            if not value:
                errors.append(f"{item.name or item.imagery_id}：{label}不能为空")
            elif VAGUE_IMAGERY_PATTERN.search(value):
                errors.append(f"{item.name or item.imagery_id}：{label}含模糊表达“{VAGUE_IMAGERY_PATTERN.search(value).group()}”")
    names = [item.name.strip().casefold() for item in selected]
    if len(names) != len(set(names)):
        errors.append("保留的意象名称不能重复")
    return errors


def imagery_prompt_text(imagery):
    """Compact fallback text; normal runs use DeepSeek's relational scene paragraph."""
    parts = [
        f"{item.quantity_description}{item.name}位于{item.position}，{item.appearance}，{item.state_action}，"
        f"并与{item.relationship}共同构成空间关系"
        for item in imagery
    ]
    return "画面中" + "；".join(parts) + "，整体作为同一场景呈现。"


def ensure_anatomy_constraint(prompt):
    if ANATOMY_CONSTRAINT in prompt:
        return prompt
    return prompt.rstrip() + f"\n\n【绝对需要避免的内容】\n{ANATOMY_CONSTRAINT}"


def direction_module_selection(project, direction_id):
    if direction_id in project.direction_module_ids:
        return normalize_module_ids(project.direction_module_ids[direction_id], content_only=True)
    direction = next((item for item in project.intent_analysis.visual_directions
                      if item.direction_id == direction_id), None) if project.intent_analysis else None
    recommended = direction.recommended_module_ids if direction else []
    if not recommended and project.intent_analysis:
        recommended = project.intent_analysis.suggested_modules
    return recommended_content_modules(recommended)


def module_term_candidates(kb, module_id, query, current_ids=None, limit=10):
    spec = MODULE_BY_ID[module_id]
    current = [kb.by_id[identity] for identity in current_ids or [] if identity in kb.by_id]
    matches = kb.search((query + " " + spec.search_hint).strip(), limit=limit,
                        category=spec.knowledge_categories) if spec.knowledge_categories else []
    seen = set()
    result = []
    for term in current + matches:
        if term["id"] not in seen:
            seen.add(term["id"])
            result.append(term)
    return result[:max(limit, len(current))]


def _preferred_module_for_term(term, selected_ids):
    name = (term["name"] + " " + term.get("english_name", "")).lower()
    if any(word in name for word in ("色", "palette", "color", "调色")) and "color" in selected_ids:
        return "color"
    preferred = {
        "art_history": "core_style", "anime": "core_style", "culture": "core_style",
        "method": "medium_appearance", "line_texture": "medium_appearance", "photo": "medium_appearance",
        "camera": "viewpoint", "composition": "composition", "graphic": "composition",
        "environment": "scene_details", "character_design": "scene_details", "pose": "scene_details",
        "ornament": "scene_details", "lighting": "lighting", "emotion": "mood", "narrative": "mood",
    }.get(term["category"])
    if preferred in selected_ids:
        return preferred
    return next((identity for identity in selected_ids
                 if term["category"] in MODULE_BY_ID[identity].knowledge_categories), None)


def initialize_direction_module_setup(project, direction_id, submitted_text, kb):
    """Seed a visible, editable module/knowledge setup once for each direction."""
    selected_ids = direction_module_selection(project, direction_id)
    if direction_id not in project.direction_module_ids and infer_detail_mode(project, direction_id, submitted_text, kb) == "clean":
        preferred_clean = ["core_style", "medium_appearance", "viewpoint", "composition", "scene_details", "color"]
        selected_ids = [identity for identity in preferred_clean if identity in selected_ids]
        for identity in preferred_clean:
            if len(selected_ids) >= MIN_CONTENT_MODULES:
                break
            if identity not in selected_ids:
                selected_ids.append(identity)
    project.direction_module_ids.setdefault(direction_id, selected_ids)
    term_map = project.direction_module_term_ids.setdefault(direction_id, {})
    note_map = project.direction_module_notes.setdefault(direction_id, {})
    direction = next(item for item in project.intent_analysis.visual_directions
                     if item.direction_id == direction_id)
    preferred_terms = list(dict.fromkeys([*direction.knowledge_term_ids, *project.selected_term_ids]))
    for identity in selected_ids:
        note_map.setdefault(identity, "")
        if identity in term_map:
            continue
        assigned = []
        for term_id in preferred_terms:
            term = kb.by_id.get(term_id)
            if term and _preferred_module_for_term(term, selected_ids) == identity:
                assigned.append(term_id)
        if not assigned:
            candidates = module_term_candidates(kb, identity,
                submitted_text + " " + " ".join(direction.technical_terms), limit=1)
            assigned = [term["id"] for term in candidates[:1]]
        term_map[identity] = assigned[:3]
    constraints = project.direction_constraint_modules.setdefault(direction_id, [])
    if (project.must_keep.strip() or project.locked_constraints.strip()) and "must_keep" not in constraints:
        constraints.append("must_keep")
    if project.must_avoid.strip() and "must_avoid" not in constraints:
        constraints.append("must_avoid")
    for identity in constraints:
        note_map.setdefault(identity, "")
    return selected_ids


def assemble_prompt(modules):
    """The final prompt is an exact ordered join of the reviewed fixed modules."""
    sections = []
    for module in modules:
        content = (module.confirmed_content or module.ai_draft).strip()
        if content:
            sections.append(f"【{module.title}】\n{content}")
    return "\n\n".join(sections)


def _module_requests(project, kb, direction_id, module_ids):
    term_map = project.direction_module_term_ids.get(direction_id, {})
    note_map = project.direction_module_notes.get(direction_id, {})
    requests = []
    for identity in module_ids:
        spec = MODULE_BY_ID[identity]
        term_ids = list(dict.fromkeys(term_map.get(identity, [])))[:3]
        requests.append({
            "id": identity,
            "title": spec.title,
            "purpose": spec.purpose,
            "module_note": note_map.get(identity, "").strip(),
            "knowledge_material": kb.prompt_material(term_ids),
        })
        if identity == "core_style":
            direction = next((item for item in project.intent_analysis.visual_directions
                              if item.direction_id == direction_id), None)
            selected_candidate_ids = set(project.direction_style_candidate_ids.get(direction_id, []))
            requests[-1]["selected_style_candidates"] = [candidate.model_dump(mode="json")
                for candidate in (direction.style_candidates if direction else [])
                if candidate.candidate_id in selected_candidate_ids]
    return requests


def _planning_context(project, direction_id, submitted_text, module_requests):
    direction = next(item for item in project.intent_analysis.visual_directions
                     if item.direction_id == direction_id)
    return {
        "initial_intent": project.initial_intent,
        "first_stage_analysis": project.intent_analysis.model_dump(mode="json"),
        "selected_direction": direction.model_dump(mode="json"),
        "submitted_direction": submitted_text.strip(),
        "selection_rule": "The submitted direction, confirmed imagery and module setup override earlier drafts.",
        "module_setup": module_requests,
    }


def plan_direction_imagery(project, deepseek, kb, direction_id, submitted_text):
    """Create the separate user-reviewable imagery inventory without drafting the final prompt."""
    if not submitted_text.strip():
        raise ValueError("方案内容不能为空")
    if not project.intent_analysis or direction_id not in {d.direction_id for d in project.intent_analysis.visual_directions}:
        raise ValueError("请先选择一个方案")
    content_ids = direction_module_selection(project, direction_id)
    if not valid_content_selection(content_ids):
        raise ValueError(f"请选择 {MIN_CONTENT_MODULES}–{MAX_CONTENT_MODULES} 个内容模块，并保留场景与细节")
    module_requests = _module_requests(project, kb, direction_id, content_ids)
    planning_context = _planning_context(project, direction_id, submitted_text, module_requests)
    detail_mode = infer_detail_mode(project, direction_id, submitted_text, kb)
    refs = [r.model_dump(mode="json") for r in project.reference_images]
    imagery = deepseek.plan_imagery(planning_context, detail_mode=detail_mode, reference_images=refs)
    errors = imagery_validation_errors(imagery)
    if errors:
        raise RuntimeError("DeepSeek意象清单不够明确：" + "；".join(errors[:3]))
    project.direction_imagery[direction_id] = imagery
    project.direction_imagery_source_text[direction_id] = submitted_text.strip()
    project.direction_imagery_detail_modes[direction_id] = detail_mode
    project.selected_direction = direction_id
    project.selected_direction_text = submitted_text.strip()
    project.direction_drafts[direction_id] = submitted_text
    return imagery


def develop_direction(project, deepseek, kb, direction_id, submitted_text):
    """Develop modules only after the user has reviewed a separate imagery inventory."""
    if not submitted_text.strip():
        raise ValueError("方案内容不能为空")
    if not project.intent_analysis or direction_id not in {d.direction_id for d in project.intent_analysis.visual_directions}:
        raise ValueError("请先选择一个方案")
    content_ids = direction_module_selection(project, direction_id)
    if not valid_content_selection(content_ids):
        raise ValueError(f"请选择 {MIN_CONTENT_MODULES}–{MAX_CONTENT_MODULES} 个内容模块，并保留场景与细节")
    imagery = project.direction_imagery.get(direction_id, [])
    detail_mode = infer_detail_mode(project, direction_id, submitted_text, kb)
    if (not imagery or project.direction_imagery_source_text.get(direction_id) != submitted_text.strip()
            or project.direction_imagery_detail_modes.get(direction_id) != detail_mode):
        raise ValueError("方案内容已变化，请先重新生成并确认画面意象")
    imagery_errors = imagery_validation_errors(imagery)
    if imagery_errors:
        raise ValueError("请先修正意象清单：" + "；".join(imagery_errors[:3]))
    confirmed_imagery = [item.model_copy(deep=True) for item in imagery if item.selected]
    excluded_imagery = [item.name for item in imagery if not item.selected]
    constraint_ids = [identity for identity in project.direction_constraint_modules.get(direction_id, [])
                      if identity in CONSTRAINT_MODULE_IDS]
    if (project.must_keep.strip() or project.locked_constraints.strip()) and "must_keep" not in constraint_ids:
        constraint_ids.append("must_keep")
    if project.must_avoid.strip() and "must_avoid" not in constraint_ids:
        constraint_ids.append("must_avoid")
    contains_people = project.mode == "text2img" and direction_contains_people(project, direction_id, submitted_text)
    if contains_people and "must_avoid" not in constraint_ids:
        constraint_ids.append("must_avoid")
    module_ids = [*content_ids, *constraint_ids]
    term_map = project.direction_module_term_ids.get(direction_id, {})
    note_map = project.direction_module_notes.get(direction_id, {})
    module_requests = _module_requests(project, kb, direction_id, module_ids)
    refs = [r.model_dump(mode="json") for r in project.reference_images]
    constraints = constraints_for(project)
    planning_context = _planning_context(project, direction_id, submitted_text, module_requests)
    planning_context["confirmed_imagery"] = [item.model_dump(mode="json") for item in confirmed_imagery]
    planning_context["detail_mode"] = detail_mode
    planning_context["detail_policy"] = detail_mode_policy(detail_mode)
    modules, diffs = deepseek.develop_prompt(planning_context, module_requests,
        constraints=constraints, reference_images=refs,
        confirmed_imagery=[item.model_dump(mode="json") for item in confirmed_imagery],
        excluded_imagery=excluded_imagery, detail_mode=detail_mode,
        anatomy_constraint=ANATOMY_CONSTRAINT if contains_people else "")
    scene_module = next((module for module in modules if module.module_id == "scene_details"), None)
    if scene_module is None:
        raise RuntimeError("DeepSeek未返回场景与细节模块，请重试")
    had_imagery_ids = bool(scene_module.imagery_ids)
    if not had_imagery_ids:
        # Compatibility with older integrations; current DeepSeek responses carry these IDs.
        scene_module.imagery_ids = [item.imagery_id for item in confirmed_imagery]
        scene_text = scene_module.confirmed_content or scene_module.ai_draft
        missing_names = [item.name for item in confirmed_imagery if item.name not in scene_text]
        if missing_names:
            scene_module.ai_draft = (scene_text.rstrip() + " " + imagery_prompt_text(confirmed_imagery)).strip()
    for name in excluded_imagery:
        if any(name and name in (module.confirmed_content or module.ai_draft)
               for module in modules if module.module_id not in CONSTRAINT_MODULE_IDS):
            raise RuntimeError(f"DeepSeek重新加入了已取消意象“{name}”，请重试")
    if contains_people:
        avoid_module = next(module for module in modules if module.module_id == "must_avoid")
        for module in modules:
            if module.module_id != "must_avoid":
                module.ai_draft = (module.ai_draft or "").replace(ANATOMY_CONSTRAINT, "").strip()
        body = (avoid_module.confirmed_content or avoid_module.ai_draft).strip()
        body = body.replace(ANATOMY_CONSTRAINT, "").strip()
        avoid_module.ai_draft = ANATOMY_CONSTRAINT if not body else ANATOMY_CONSTRAINT + "\n" + body
    prompt = assemble_prompt(modules)
    if contains_people:
        prompt = ensure_anatomy_constraint(prompt)
    if not prompt.strip():
        raise ValueError("DeepSeek 未生成最终提示词，请重试")
    previous = current_iteration(project)
    iteration = Iteration(revision=next_revision(project), iteration_type="initial" if not previous else "redraw",
        locked_prompt=prompt, prompt_modules=modules, prompt_diff=diffs,
        direction_id=direction_id, direction_text=submitted_text.strip(), constraints_snapshot=constraints,
        prompt_module_ids=module_ids,
        module_term_ids={identity: list(term_map.get(identity, [])) for identity in module_ids},
        module_notes={identity: note_map.get(identity, "") for identity in module_ids},
        planning_snapshot=planning_context,
        confirmed_imagery=confirmed_imagery, detail_mode=detail_mode, contains_people=contains_people,
        reference_images=[r.model_copy(deep=True) for r in project.reference_images],
        aspect_ratio=project.draft_aspect_ratio or (previous.aspect_ratio if previous else "1:1"),
        interaction_model=deepseek.model, knowledge_version=SCHEMA_VERSION)
    iteration.knowledge_term_ids = list(dict.fromkeys(
        identity for module_id in module_ids for identity in term_map.get(module_id, []) if identity in kb.by_id))
    direction = next(
        item for item in project.intent_analysis.visual_directions
        if item.direction_id == direction_id
    )
    selected_style_ids = set(project.direction_style_candidate_ids.get(direction_id, []))
    selected_style_term_ids = [candidate.term_id for candidate in direction.style_candidates
                               if candidate.candidate_id in selected_style_ids and candidate.term_id in kb.by_id]
    if selected_style_term_ids and "core_style" in iteration.module_term_ids:
        iteration.module_term_ids["core_style"] = list(dict.fromkeys(
            iteration.module_term_ids["core_style"] + selected_style_term_ids))
    iteration.knowledge_term_ids.extend(
        candidate.term_id for candidate in direction.style_candidates
        if candidate.candidate_id in selected_style_ids and candidate.term_id in kb.by_id
        and candidate.term_id not in iteration.knowledge_term_ids)
    attach_iteration(project, iteration)
    project.selected_direction = direction_id
    project.selected_direction_text = submitted_text.strip()
    project.direction_drafts[direction_id] = submitted_text
    project.prompt_draft = prompt
    project.prompt_draft_version += 1
    project.status = "in_progress"
    return iteration


def prepare_generation(project, submitted_prompt, ratio, size):
    """Freeze the editor text, restoring the mandatory anatomy guard for people when needed."""
    if not submitted_prompt.strip():
        raise ValueError("最终提示词不能为空")
    source = current_iteration(project)
    if source is None:
        raise ValueError("请先提交方案，让 DeepSeek 深化提示词")
    guarded_prompt = submitted_prompt
    if project.mode == "text2img" and (source.contains_people or prompt_contains_people(submitted_prompt)):
        guarded_prompt = ensure_anatomy_constraint(submitted_prompt)
    target = source
    if source.outputs or source.api_calls or source.model_name or source.error_message:
        target = fork_iteration(project, source, "redraw", prompt=guarded_prompt)
        attach_iteration(project, target)
    target.locked_prompt = guarded_prompt
    target.contains_people = project.mode == "text2img" and (
        source.contains_people or prompt_contains_people(guarded_prompt))
    target.aspect_ratio, target.output_size = ratio, size
    target.needs_integration = False
    project.prompt_draft = guarded_prompt
    project.draft_aspect_ratio = ratio
    project.status = "in_progress"
    return target


def record_outputs(iteration, image_paths, api_calls):
    iteration.api_calls += api_calls
    for index, path in enumerate(image_paths):
        output = GenerationOutput(image_path=path, slot="A" if index % 2 == 0 else "B")
        if Path(path).exists():
            try:
                with Image.open(path) as image:
                    output.actual_width, output.actual_height = image.size
            except OSError:
                pass
        iteration.outputs.append(output)
    iteration.success = len(image_paths) == 2
    iteration.error_message = "" if iteration.success else "只完成部分图片，已保留，可继续生成"
