"""Opt-in live verification. Offline tests never invoke paid APIs.
Run: python test_deepseek.py --live [--images]
"""
import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from api_clients import DeepSeekClient, OpenAIImageClient
from knowledge_base import KnowledgeBase, SCHEMA_VERSION
from storage import ROOT, WORKSPACE, atomic_json
from models import Project, IntentAnalysis, SceneImagery
from workflow import (develop_direction, direction_content, initialize_direction_module_setup,
                      prepare_generation, plan_direction_imagery)


def verify_creation_workflow(client, kb):
    """Three real planning calls; drawing submission is checked without paid image calls."""
    intent = "干净的现代轻小说插画：清晨有雾的乡间车站，一位穿蓝色外套的少女在右侧等车，背景简单，不要粒子、泛光和复杂纹理。"
    report = {"at": datetime.now(timezone.utc).isoformat(), "model": client.model,
              "reasoning_effort": client.reasoning_effort, "intent": intent}
    analysis = client.analyze_intent(intent)
    report["analysis"] = analysis.model_dump(mode="json")
    if any(not d.expanded_idea.strip() for d in analysis.visual_directions):
        raise RuntimeError("DeepSeek must return editable expanded ideas")
    if any(len(d.style_candidates) < 5 or any(candidate.selected for candidate in d.style_candidates)
           for d in analysis.visual_directions):
        raise RuntimeError("DeepSeek must return at least five unselected style candidates per direction")
    print("EXPANDED_IDEAS_OK", len(analysis.visual_directions), flush=True)
    project = Project(project_id="workflow_verification", short_name="流程验证", display_name="流程验证",
        initial_intent=intent, intent_analysis=analysis, must_avoid="可读文字、签名、水印")
    direction = analysis.visual_directions[0]
    edited = direction_content(direction) + "\n\n用户修改：将旅人的外套改成红色，保持画面右侧的位置；不再使用蓝色外套。"
    report["submitted_direction"] = edited
    initialize_direction_module_setup(project, direction.direction_id, edited, kb)
    imagery = plan_direction_imagery(project, client, kb, direction.direction_id, edited)
    report["imagery"] = [item.model_dump(mode="json") for item in imagery]
    if not any(item.optional_candidate and not item.selected for item in imagery):
        raise RuntimeError("Imagery plan must include at least one unselected optional candidate")
    print("IMAGERY_OK", len(imagery), flush=True)
    iteration = develop_direction(project, client, kb, direction.direction_id, edited)
    report["developed_iteration"] = iteration.model_dump(mode="json")
    atomic_json(WORKSPACE / "verification" / "workflow_live.json", report)
    titles = [m.title for m in iteration.prompt_modules]
    if titles[-1] not in ("绝对需要遵守的内容", "绝对需要避免的内容") or any(
            f"【{title}】" not in iteration.locked_prompt for title in titles):
        raise RuntimeError("Final prompt must retain all editable section headings")
    if not any(token in iteration.locked_prompt for token in ("红色", "红外套", "红衣")):
        raise RuntimeError("Edited direction was not reflected in the final prompt")
    if "×" in iteration.locked_prompt or "img_" in iteration.locked_prompt:
        raise RuntimeError("Final prompt must use a coherent scene paragraph, not imagery IDs or multiplication notation")
    if not 6 <= len([identity for identity in iteration.prompt_module_ids if not identity.startswith("must_")]) <= 8:
        raise RuntimeError("First-stage module selection must contain 6-8 fixed content modules")
    if not iteration.knowledge_term_ids:
        raise RuntimeError("No visible knowledge terms were carried into prompt development")
    print("DEVELOPMENT_OK", len(titles), "sections", len(iteration.locked_prompt), "characters", flush=True)
    final_edit = iteration.locked_prompt + "\n\n本轮手动修改：左侧雾中保留更多留白。"
    target = prepare_generation(project, final_edit, "16:9", "3840x2160")
    if target.locked_prompt != final_edit:
        raise RuntimeError("Drawing snapshot must exactly match the editor text")
    report["submitted_prompt"] = target.locked_prompt
    report["planning_calls"] = 3
    report["image_calls"] = 0
    report["success"] = True
    atomic_json(WORKSPACE / "verification" / "workflow_live.json", report)
    print("WORKFLOW_LIVE_OK", client.model, flush=True)


def verify_concise_workflow(client, kb):
    """Reuse the previous scene to compare length and check feedback stays concise."""
    baseline = json.loads((WORKSPACE / "verification" / "workflow_live.json").read_text(encoding="utf-8"))
    project = Project(project_id="concise_verification", short_name="精炼验证", display_name="精炼验证",
        initial_intent=baseline["intent"], intent_analysis=IntentAnalysis.model_validate(baseline["analysis"]),
        must_avoid="可读文字、签名、水印")
    report = {"at": datetime.now(timezone.utc).isoformat(), "model": client.model,
        "reasoning_effort": client.reasoning_effort, "baseline_length": len(baseline["developed_iteration"]["locked_prompt"]),
        "submitted_direction": baseline["submitted_direction"], "image_calls": 0, "success": False}
    target_path = WORKSPACE / "verification" / "concise_workflow_live.json"
    direction = next(item for item in project.intent_analysis.visual_directions
                     if item.direction_id == baseline["developed_iteration"]["direction_id"])
    initialize_direction_module_setup(project, direction.direction_id, baseline["submitted_direction"], kb)
    developed = baseline["developed_iteration"]
    if developed.get("confirmed_imagery"):
        project.direction_imagery[direction.direction_id] = [SceneImagery.model_validate(item)
                                                              for item in developed["confirmed_imagery"]]
        project.direction_imagery_source_text[direction.direction_id] = baseline["submitted_direction"]
        project.direction_detail_modes[direction.direction_id] = developed.get("detail_mode", "balanced")
        project.direction_imagery_detail_modes[direction.direction_id] = developed.get("detail_mode", "balanced")
    else:
        plan_direction_imagery(project, client, kb, direction.direction_id, baseline["submitted_direction"])
    iteration = develop_direction(project, client, kb, baseline["developed_iteration"]["direction_id"],
                                  baseline["submitted_direction"])
    report["developed_iteration"] = iteration.model_dump(mode="json")
    report["prompt_length"] = len(iteration.locked_prompt)
    atomic_json(target_path, report)

    def check_prompt(prompt):
        positions = [prompt.find(f"【{m.title}】") for m in iteration.prompt_modules]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise RuntimeError("Section headings must remain complete and ordered")
        if len(prompt) > 900:
            raise RuntimeError(f"Simple-scene prompt remains too long: {len(prompt)} characters")
        if not all(term in prompt for term in ("红", "右", "轻小说", "文字", "水印")):
            raise RuntimeError("Concise prompt lost a key subject, style or explicit constraint")

    check_prompt(iteration.locked_prompt)
    print("CONCISE_DEVELOPMENT_OK", report["baseline_length"], "->", report["prompt_length"], flush=True)
    diagnosis, diffs, revised = client.analyze_feedback(iteration.locked_prompt,
        "背景再简化，左侧保留更多留白，提示词继续简洁，不扩写细枝末节。",
        "少女红色外套、位于右侧、干净轻小说插画、不要可读文字与水印。", iteration.prompt_modules,
        "以当前最终稿的明确约束为准。")
    report["feedback"] = {"diagnosis": diagnosis, "diffs": [d.model_dump(mode="json") for d in diffs],
                          "prompt": revised, "prompt_length": len(revised)}
    atomic_json(target_path, report)
    check_prompt(revised)
    report["success"] = True
    atomic_json(target_path, report)
    print("CONCISE_FEEDBACK_OK", len(revised), "characters; no image calls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly call DeepSeek API")
    parser.add_argument("--images", action="store_true", help="Also generate one 1024px low-quality verification image")
    parser.add_argument("--workflow", action="store_true", help="Verify the revised editable creation flow, no image calls")
    parser.add_argument("--concise", action="store_true", help="Compare concise planning against the saved workflow scene, no image calls")
    args = parser.parse_args()
    if not args.live:
        print("No API calls made. Use --live for actual verification.")
        return
    load_dotenv(ROOT / ".env")
    kb = KnowledgeBase.load()
    client = DeepSeekClient(os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_BASE_URL"),
        os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        os.getenv("DEEPSEEK_REASONING_EFFORT", "medium"), knowledge=kb,
        use_env_proxy=os.getenv("DEEPSEEK_USE_ENV_PROXY", "false").lower() == "true")
    if args.concise:
        if args.images or args.workflow:
            parser.error("--concise is a separate text-only verification")
        verify_concise_workflow(client, kb)
        return
    if args.workflow:
        if args.images:
            parser.error("--workflow does not generate images; use --live --images for the separate image smoke test")
        verify_creation_workflow(client, kb)
        return
    available = client.client.models.list().model_dump()
    ids = {model["id"] for model in available["data"]}
    if client.model not in ids:
        raise RuntimeError("Configured interaction model is not in the current official model list")
    report = {"at": datetime.now(timezone.utc).isoformat(), "model": client.model,
              "reasoning_effort": client.reasoning_effort, "official_models": available,
              "knowledge_version": SCHEMA_VERSION, "knowledge_count": len(kb.terms), "scenarios": []}
    scenes = [
        "清晨有雾的乡间车站，一位穿蓝色外套的旅人站在右侧等车，温暖水彩插画，保留主体清晰，背景不要复杂。",
        "构成主义海报，红黑米白几何块面，工人剪影与齿轮，非对称版式，不要文字。",
        "赛博朋克雨夜街道，橘猫抱着小水獭站在霓虹招牌下，低机位，中景，温暖而孤独。",
    ]
    analyses = []
    for index, scene in enumerate(scenes, 1):
        analysis = client.analyze_intent(scene)
        analyses.append(analysis)
        report["scenarios"].append({"intent": scene, "analysis": analysis.model_dump(mode="json")})
        atomic_json(WORKSPACE / "verification" / "deepseek_live.json", report)
        print("INTENT_OK", index, len(analysis.visual_directions), flush=True)
    direction = analyses[0].visual_directions[0]
    modules = client.generate_module_drafts(scenes[0], direction.name + direction.visual_consequences,
                                            analyses[0].suggested_modules)
    if not any(m.knowledge_term_ids for m in modules):
        raise RuntimeError("No grounded term references in generated modules")
    report["modules"] = [m.model_dump(mode="json") for m in modules]
    print("MODULES_OK", len(modules), flush=True)
    rewritten = client.revise_module(modules[0], "保留主体，把背景的细节数量减少，画面更宁静。",
        "\n".join(m.title + ":" + m.ai_draft for m in modules[1:]),
        "必须保留蓝色外套；不要文字与水印")
    report["rewritten_module"] = rewritten
    for module in modules:
        module.confirmed_content = module.ai_draft
    prompt, conflicts = client.integrate_prompt(modules, "必须保留蓝色外套；不要文字与水印")
    report["integrated_prompt"] = prompt
    report["conflicts"] = [d.model_dump() for d in conflicts]
    print("INTEGRATE_OK", len(prompt), flush=True)
    diagnosis, diffs, new_prompt = client.analyze_feedback(prompt, "背景太复杂，减少站台设施与远处细节",
        "保持主体位置和蓝色外套", modules, "不要文字与水印")
    report["feedback"] = {"diagnosis": diagnosis, "diffs": [d.model_dump() for d in diffs], "new_prompt": new_prompt}
    report["balance_query_success"] = bool(client.get_balance().get("is_available"))
    if args.images:
        images = OpenAIImageClient(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
            os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"), quality="low",
            response_format=os.getenv("OPENAI_IMAGE_RESPONSE_FORMAT", "url"),
            use_env_proxy=False, output_dir=str(WORKSPACE / "verification" / "images"))
        paths = images.generate_images(prompt, num_images=1, resolution="1024x1024")
        from PIL import Image
        with Image.open(paths[0]) as image:
            report["image_verification"] = {"path": paths[0], "actual_size": list(image.size), "requested_size": "1024x1024"}
        print("IMAGE_OK", flush=True)
    report["success"] = True
    atomic_json(WORKSPACE / "verification" / "deepseek_live.json", report)
    print("LIVE_VERIFICATION_OK", client.model, flush=True)


if __name__ == "__main__":
    main()
