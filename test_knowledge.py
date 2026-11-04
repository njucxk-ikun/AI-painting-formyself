"""Offline regression tests for source coverage, search, checkpoints and prompt grounding."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI

from deepseek_client import DeepSeekClient
from knowledge_base import KnowledgeBase, KnowledgeTerm
from models import DynamicModule, Project, Iteration, PromptDiff, SceneImagery, IntentAnalysis, VisualDirection, StyleCandidate
from prompt_catalog import ANATOMY_CONSTRAINT, recommended_content_modules
from storage import atomic_json
import workflow


def seed_term(identifier="sample", name="测试术语"):
    # Test-only placeholders; never published as knowledge content.
    return {
        "id": identifier, "name": name, "category": "test", "category_name": "测试分类", "english_name": "test term",
        "aliases": [], "nature": "测试", "description": "结构验证使用的占位文本，不代表视觉术语解释，也不会被写入发布的知识库。" * 2,
        "visual_effect": "仅用于本地回归测试，验证检索和序列化逻辑，不会用于真实绘画提示词。",
        "visual_cues": ["测试要点甲", "测试要点乙", "测试要点丙"], "applicable_scenes": ["测试场景甲", "测试场景乙"],
        "avoid_when": ["测试边界"], "prompt_keywords": ["test alpha", "test beta", "test gamma"], "examples": ["本地测试示例"],
        "confused_with": [{"term": "关联测试词", "note": "仅用来检查易混淆结构。"}],
        "combinations": [{"term": "关联测试词", "note": "仅用来检查组合关系结构。"}], "cautions": ["仅用于测试"],
        "source": "unittest", "provenance": {"provider": "test", "model": "fake"},
    }


class KnowledgeTests(unittest.TestCase):
    def test_exact_alias_and_descriptive_search(self):
        terms = [seed_term("a", "构成主义"), seed_term("b", "水彩")]
        terms[0]["aliases"] = ["Constructivism"]
        terms[1]["visual_cues"] = ["水分在纸面渗透", "透明叠色", "柔和晕染"]
        kb = KnowledgeBase(terms)
        self.assertEqual(kb.search("Constructivism")[0]["id"], "a")
        self.assertEqual(kb.search("柔和晕染")[0]["id"], "b")
        self.assertEqual(kb.search("never-matches-xyz"), [])

    def test_selected_terms_prioritized_and_context_budget(self):
        terms = [seed_term(str(i), "术语" + str(i)) for i in range(20)]
        kb = KnowledgeBase(terms)
        context, ids = kb.context("术语0", ["19"], max_chars=1500)
        self.assertEqual(ids[0], "19")
        self.assertLessEqual(len(context), 1500)
        self.assertIn('"selected_by_user": true', context)
        full, _ = kb.context("术语0", limit=2, max_chars=100000)
        tight, _ = kb.context("术语0", limit=2, max_chars=len(full) - 1)
        self.assertLessEqual(len(tight), len(full) - 1)

    def test_invalid_content_rejected(self):
        invalid = seed_term()
        invalid["description"] = "过短"
        with self.assertRaises(ValueError):
            KnowledgeTerm.model_validate(invalid)

    def test_published_knowledge_has_valid_relations(self):
        kb = KnowledgeBase.load()
        self.assertGreater(len(kb.terms), 1000)
        for term in kb.terms:
            for relation in term["confused_with"] + term["combinations"]:
                self.assertTrue(relation["term_id"] is None or relation["term_id"] in kb.by_id)


class DeepSeekTests(unittest.TestCase):
    def test_balance_parses_with_real_sdk_transport(self):
        def handler(request):
            self.assertEqual(request.url.path, "/user/balance")
            return httpx.Response(200, json={"is_available": True, "balance_infos": []})
        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            sdk = OpenAI(api_key="test", base_url="https://api.deepseek.com", http_client=transport)
            client = DeepSeekClient("test", client=sdk, knowledge=KnowledgeBase([]))
            self.assertTrue(client.get_balance()["is_available"])

    def make_client(self, contents, finish="stop"):
        self.calls = []
        def create(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=contents.pop(0)), finish_reason=finish)])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return DeepSeekClient("test", model="deepseek-flash", knowledge=KnowledgeBase([seed_term()]), client=sdk)

    def test_json_mode_thinking_and_structural_retry(self):
        client = self.make_client(["{invalid", json.dumps({"integrated_prompt": "完整测试提示词，包含主体构图光影等信息并验证结构化响应的正确解析。", "conflicts": []})])
        prompt, diffs = client.integrate_prompt([DynamicModule(module_id="m", title="构图", order=1, ai_draft="测试")], "保留约束")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(self.calls[0]["reasoning_effort"], "high")
        self.assertIn("保留约束", self.calls[0]["messages"][1]["content"])

    def test_truncated_response_does_not_parse(self):
        client = self.make_client(["{}"], finish="length")
        with self.assertRaisesRegex(RuntimeError, "完整"):
            client._call_api("test", "test", "json")

    def test_retrieval_uses_relevant_term_beyond_first_five(self):
        client = self.make_client(["修改后的测试模块"])
        terms = [seed_term(str(i), "占位词" + str(i)) for i in range(8)]
        terms[-1]["name"] = "伦勃朗光"
        client.knowledge = KnowledgeBase(terms)
        client.revise_module(DynamicModule(module_id="m", title="光影", order=1), "改为伦勃朗光", "", "")
        self.assertIn("伦勃朗光", self.calls[0]["messages"][1]["content"])
        self.assertIn('\\"id\\": \\"7\\"'.replace('\\\\', '\\'), self.calls[0]["messages"][1]["content"])

    def test_fixed_module_development_receives_linked_context_and_exact_knowledge(self):
        response = {"modules": [
            {"module_id": "core_style", "title": "核心风格", "order": 8,
             "ai_draft": "构成主义视觉语言，强调非对称几何秩序。", "knowledge_term_ids": ["sample", "hidden"]},
            {"module_id": "scene_details", "title": "场景与细节", "order": 9,
             "ai_draft": "环境背景：简洁站台。主体角色：红衣旅人。动作与神态：安静等车。",
             "knowledge_term_ids": []},
        ], "conflicts": []}
        client = self.make_client([json.dumps(response, ensure_ascii=False)])
        requests = [
            {"id": "core_style", "title": "核心风格", "purpose": "测试", "module_note": "保留几何关系",
             "knowledge_material": [{"id": "sample", "name": "构成主义", "prompt_keywords": ["constructivism"]}]},
            {"id": "scene_details", "title": "场景与细节", "purpose": "测试", "module_note": "",
             "knowledge_material": []},
        ]
        planning = {"initial_intent": "车站旅人", "submitted_direction": "红衣旅人在右侧等车",
                    "first_stage_analysis": {"restated_goal": "车站场景"}}
        modules, conflicts = client.develop_prompt(planning, requests, "不要文字")
        self.assertFalse(conflicts)
        self.assertEqual([m.module_id for m in modules], ["core_style", "scene_details"])
        self.assertEqual([m.order for m in modules], [1, 2])
        self.assertEqual(modules[0].knowledge_term_ids, ["sample"])
        self.assertIn("submitted_direction", self.calls[0]["messages"][1]["content"])
        self.assertIn("constructivism", self.calls[0]["messages"][1]["content"])
        self.assertIn("8K", self.calls[0]["messages"][0]["content"])

    def test_imagery_planning_is_separate_and_clean_mode_is_explicit(self):
        response = {"imagery": [{
            "imagery_id": "temporary", "category": "character", "name": "少女", "exact_count": 1,
            "quantity_description": "一名独立人物", "quantity_mode": "exact",
            "position": "画面中央偏右", "appearance": "白衬衫与深色短发，轮廓清楚",
            "state_action": "坐在窗边阅读", "relationship": "双手扶住一本合上的书",
            "importance": "primary", "selected": True, "reason": "唯一主体",
        }, {
            "imagery_id": "optional", "category": "environment", "name": "倒挂公寓楼", "exact_count": None,
            "quantity_description": "一整栋连续的老旧水泥公寓楼", "quantity_mode": "aggregate",
            "position": "画面左上方垂直向下", "appearance": "密集窗户、空调外机、金属管道和防盗网",
            "state_action": "以侧面边缘倒挂在画面中", "relationship": "作为主体周围的压迫性背景",
            "importance": "secondary", "optional_candidate": True, "selected": True, "reason": "可选背景结构",
        }]}
        client = self.make_client([json.dumps(response, ensure_ascii=False)])
        imagery = client.plan_imagery({"submitted_direction": "轻小说插画中的窗边少女"}, detail_mode="clean")
        self.assertEqual(imagery[0].imagery_id, "img_001")
        self.assertFalse(imagery[1].selected)
        self.assertEqual(imagery[1].quantity_description, "一整栋连续的老旧水泥公寓楼")
        payload = json.loads(self.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["detail_mode"], "clean")
        self.assertIn("不要添加装饰粒子", payload["detail_policy"])
        self.assertIn("模棱两可", self.calls[0]["messages"][0]["content"])

    def test_style_candidates_are_optional_and_not_auto_selected(self):
        response = {"restated_goal": "荒诞职场照片", "main_challenges": [], "suggested_modules": ["core_style"],
                "visual_directions": [{
                "direction_id": "d", "name": "荒诞摄影", "technical_terms": [],
                "explanation": "测试", "visual_consequences": "测试", "applicable_reason": "测试",
                "is_default": True, "recommended_module_ids": ["core_style", "medium_appearance",
                    "viewpoint", "composition", "scene_details", "lighting"], "knowledge_term_ids": [],
                "style_candidates": [{"candidate_id": "old", "name": "超现实主义摄影",
                    "english_name": "Surrealist Photography", "term_id": "sample",
                    "prompt_phrase": "surrealist photography", "rationale": "测试", "selected": True}]
            }, {
                "direction_id": "d2", "name": "另一方向", "technical_terms": [],
                "explanation": "测试", "visual_consequences": "测试", "applicable_reason": "测试",
                "recommended_module_ids": ["core_style", "medium_appearance", "viewpoint", "composition", "scene_details", "lighting"],
                "knowledge_term_ids": []
            }]}
        client = self.make_client([json.dumps(response, ensure_ascii=False)])
        result = client.analyze_intent("荒诞职场照片")
        candidate = result.visual_directions[0].style_candidates[0]
        self.assertFalse(candidate.selected)
        self.assertEqual(candidate.candidate_id, "style_001")

    def test_selected_style_candidates_are_sent_only_to_core_style(self):
        project = Project(project_id="style", short_name="style", display_name="style", initial_intent="x",
            intent_analysis=IntentAnalysis(restated_goal="x", main_challenges=[], visual_directions=[
                VisualDirection(direction_id="d", name="d", technical_terms=[], explanation="x",
                    visual_consequences="x", applicable_reason="x", style_candidates=[
                        StyleCandidate(candidate_id="style_001", name="超现实主义摄影", prompt_phrase="surrealist photography")])]))
        project.direction_style_candidate_ids["d"] = ["style_001"]
        project.direction_module_term_ids["d"] = {"core_style": []}
        requests = workflow._module_requests(project, KnowledgeBase([]), "d", ["core_style", "lighting"])
        self.assertEqual(requests[0]["selected_style_candidates"][0]["name"], "超现实主义摄影")
        self.assertEqual(requests[1].get("selected_style_candidates", []), [])


class WorkflowTests(unittest.TestCase):
    def project(self):
        return Project(project_id="sample", short_name="测试", display_name="测试", initial_intent="test")

    def test_legacy_module_names_become_fixed_six_to_eight_selection(self):
        selected = recommended_content_modules(["概念与风格", "构图与视角", "场景与细节", "光影与氛围"])
        self.assertGreaterEqual(len(selected), 6)
        self.assertLessEqual(len(selected), 8)
        self.assertIn("scene_details", selected)
        self.assertTrue(set(selected) <= {"core_style", "medium_appearance", "viewpoint", "composition",
                                          "scene_details", "lighting", "color", "mood"})

    def test_feedback_copy_preserves_size_refs_and_history(self):
        project = self.project()
        original = Iteration(revision="r005", iteration_type="initial", locked_prompt="原提示词",
            prompt_modules=[DynamicModule(module_id="m", title="光影", order=1, confirmed_content="原光影")],
            aspect_ratio="16:9", output_size="3840x2160")
        workflow.attach_iteration(project, original)
        diff = PromptDiff(module="光影", action="replace", new_content="柔和光影")
        new = workflow.fork_iteration(project, original, "feedback", "新提示词", [diff])
        self.assertEqual(new.revision, "r006")
        self.assertEqual(new.aspect_ratio, "16:9")
        self.assertEqual(new.output_size, "3840x2160")
        self.assertEqual(new.prompt_modules[0].confirmed_content, "柔和光影")
        self.assertEqual(original.prompt_modules[0].confirmed_content, "原光影")
        self.assertEqual(original.locked_prompt, "原提示词")

    def test_copy_does_not_clear_stale_prompt_flag(self):
        project = self.project()
        original = Iteration(revision="r001", iteration_type="initial", locked_prompt="old", prompt_modules=[], needs_integration=True)
        workflow.attach_iteration(project, original)
        self.assertTrue(workflow.fork_iteration(project, original, "redraw").needs_integration)

    def test_corrupt_project_does_not_prevent_loading_healthy_one(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(workflow, "PROJECTS", Path(directory)):
            workflow.save_project(self.project())
            atomic_json(Path(directory) / "broken" / "project.json", {"not": "a project"})
            projects, errors = workflow.load_projects()
            self.assertEqual([p.project_id for p in projects], ["sample"])
            self.assertEqual(errors, ["broken"])

    def test_project_path_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):
            workflow.project_path("../../../outside")
        self.assertNotIn("/", workflow.new_project_id("name/with:invalid*chars"))

    def test_people_text2img_always_restores_anatomy_constraint(self):
        project = self.project()
        iteration = Iteration(revision="r001", iteration_type="initial", locked_prompt="一位少女站在窗边",
            prompt_modules=[], contains_people=True)
        workflow.attach_iteration(project, iteration)
        target = workflow.prepare_generation(project, "一位少女站在窗边", "1:1", "1024x1024")
        self.assertIn(ANATOMY_CONSTRAINT, target.locked_prompt)
        self.assertEqual(project.prompt_draft, target.locked_prompt)

    def test_nonhuman_text2img_does_not_add_anatomy_constraint(self):
        project = self.project()
        iteration = Iteration(revision="r001", iteration_type="initial", locked_prompt="一只橘猫坐在窗边", prompt_modules=[])
        workflow.attach_iteration(project, iteration)
        target = workflow.prepare_generation(project, "一只橘猫坐在窗边", "1:1", "1024x1024")
        self.assertNotIn(ANATOMY_CONSTRAINT, target.locked_prompt)

    def test_anime_direction_defaults_to_clean_detail_mode(self):
        project = self.project()
        project.intent_analysis = IntentAnalysis(restated_goal="轻小说角色", main_challenges=[], suggested_modules=[],
            visual_directions=[VisualDirection(direction_id="d", name="轻小说插画", technical_terms=["干净线稿"],
                explanation="测试说明", visual_consequences="清爽画面", applicable_reason="测试")])
        self.assertEqual(workflow.infer_detail_mode(project, "d", "二次元少女的日常场景"), "clean")

    def test_only_confirmed_imagery_reaches_prompt_development(self):
        class Planner:
            model = "fake"
            def develop_prompt(self, planning_context, module_requests, **kwargs):
                self.kwargs = kwargs
                self.module_requests = module_requests
                modules = [DynamicModule(module_id=item["id"], title=item["title"], order=index,
                    ai_draft="清晰简洁的画面") for index, item in enumerate(module_requests, 1)]
                return modules, []

        project = self.project()
        project.intent_analysis = IntentAnalysis(restated_goal="角色场景", main_challenges=[], suggested_modules=[],
            visual_directions=[VisualDirection(direction_id="d", name="测试", technical_terms=[],
                explanation="测试说明", visual_consequences="测试结果", applicable_reason="测试", contains_people=True,
                style_candidates=[
                    StyleCandidate(candidate_id="style_001", name="选中风格", term_id="selected_style",
                                   prompt_phrase="选中风格特征"),
                    StyleCandidate(candidate_id="style_002", name="未选风格", term_id="unselected_style",
                                   prompt_phrase="未选风格特征"),
                ])])
        project.direction_module_ids["d"] = ["core_style", "viewpoint", "composition", "scene_details", "lighting", "color"]
        project.direction_style_candidate_ids["d"] = ["style_001"]
        project.direction_detail_modes["d"] = "clean"
        project.direction_imagery_detail_modes["d"] = "clean"
        project.direction_imagery_source_text["d"] = "一位旅人在车站"
        project.direction_imagery["d"] = [
            SceneImagery(imagery_id="img_001", category="character", name="旅人", exact_count=1,
                quantity_description="一名独立人物", quantity_mode="exact",
                position="画面右侧", appearance="红色外套", state_action="安静站立", relationship="站在月台上"),
            SceneImagery(imagery_id="img_002", category="object", name="广告牌", exact_count=1,
                quantity_description="一块独立广告牌", quantity_mode="exact",
                position="画面左侧", appearance="蓝色矩形", state_action="固定竖立", relationship="位于旅人左侧", selected=False),
        ]
        planner = Planner()
        kb = KnowledgeBase([seed_term("selected_style"), seed_term("unselected_style")])
        iteration = workflow.develop_direction(project, planner, kb, "d", "一位旅人在车站")
        self.assertEqual([item["name"] for item in planner.kwargs["confirmed_imagery"]], ["旅人"])
        self.assertEqual(planner.kwargs["excluded_imagery"], ["广告牌"])
        self.assertIn("一名独立人物旅人位于画面右侧", iteration.locked_prompt)
        self.assertNotIn("广告牌", iteration.locked_prompt)
        self.assertNotIn("×", iteration.locked_prompt)
        self.assertIn(ANATOMY_CONSTRAINT, iteration.locked_prompt)
        self.assertEqual([candidate["candidate_id"] for candidate in planner.module_requests[0]["selected_style_candidates"]],
                         ["style_001"])
        self.assertIn("selected_style", iteration.module_term_ids["core_style"])
        self.assertIn("selected_style", iteration.knowledge_term_ids)
        self.assertNotIn("unselected_style", iteration.knowledge_term_ids)


if __name__ == "__main__":
    unittest.main()
