"""Exercise the actual three-step Streamlit UI with isolated data and fake APIs."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from api_clients import OpenAIImageClient, ImageGenerationError
from knowledge_base import KnowledgeBase
from models import DynamicModule, IntentAnalysis, VisualDirection, PromptDiff, Project, Iteration, SceneImagery
from storage import ROOT, atomic_json
from test_knowledge import seed_term
from test_openai_image import FakeImages
import workflow


class FakePlanner:
    calls = []
    fail_integration = False

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.reasoning_effort = kwargs["reasoning_effort"]

    def analyze_intent(self, intent, *args, **kwargs):
        self.calls.append(("analyze", intent))
        return IntentAnalysis(restated_goal="清晨车站的旅人", main_challenges=["保留主体，简化背景"],
            suggested_modules=["core_style", "medium_appearance", "viewpoint", "composition",
                               "scene_details", "lighting", "color", "mood"],
            visual_directions=[
                VisualDirection(direction_id="dir_1", name="方案甲", technical_terms=["测试词"],
                    expanded_idea="暖色水彩的清晨车站，蓝色外套旅人站在右侧，背景以雾气简化。",
                    explanation="水彩方向说明", visual_consequences="柔和的暖色雾气",
                    applicable_reason="符合测试", is_default=True,
                    contains_people=True,
                    recommended_module_ids=["core_style", "medium_appearance", "viewpoint", "composition",
                                            "scene_details", "lighting", "color", "mood"],
                    knowledge_term_ids=["sample"]),
                VisualDirection(direction_id="dir_2", name="方案乙", technical_terms=[],
                    expanded_idea="冷蓝色平面海报，旅人剪影居中，天空和站台形成几何色块。",
                    explanation="平面方向说明", visual_consequences="几何硬边",
                    applicable_reason="另一个视觉方案",
                    contains_people=True,
                    recommended_module_ids=["core_style", "medium_appearance", "viewpoint", "composition",
                                            "scene_details", "lighting", "color", "mood"],
                    knowledge_term_ids=["sample"])])

    def plan_imagery(self, planning_context, detail_mode="balanced", reference_images=None):
        self.calls.append(("imagery", planning_context, detail_mode, reference_images))
        return [
            SceneImagery(imagery_id="img_001", category="character", name="旅人", exact_count=1,
                quantity_description="一名独立人物", quantity_mode="exact",
                position="画面右侧三分之一", appearance="穿红色外套，轮廓清楚",
                state_action="面向轨道安静站立", relationship="站在车站月台上", importance="primary",
                reason="承载画面主体"),
            SceneImagery(imagery_id="img_002", category="environment", name="乡间车站", exact_count=1,
                quantity_description="一处简洁的乡间车站整体", quantity_mode="aggregate",
                position="画面中景和背景", appearance="简洁月台与低矮站房",
                state_action="静止在清晨薄雾中", relationship="围绕旅人形成环境", importance="secondary",
                reason="交代地点"),
        ]

    def develop_prompt(self, planning_context, module_requests, constraints="", reference_images=None, **kwargs):
        self.calls.append(("develop", planning_context, module_requests, constraints, reference_images))
        if self.fail_integration:
            raise RuntimeError("模拟深化失败，原内容保留")
        modules = []
        for index, request in enumerate(module_requests, 1):
            material_ids = [term["id"] for term in request.get("knowledge_material", [])]
            content = planning_context["submitted_direction"] if index == 1 else "保持柔和晨光，背景减少细节。"
            if request["id"] == "scene_details":
                content = "环境背景：简洁车站。主体角色：旅人。动作与神态：安静等车。"
            if request["id"] in ("must_keep", "must_avoid"):
                content = constraints
            modules.append(DynamicModule(module_id=request["id"], title=request["title"], order=index,
                ai_draft=content, knowledge_term_ids=material_ids))
        return modules, []

    def analyze_feedback(self, current_prompt, *args, **kwargs):
        self.calls.append(("feedback", current_prompt))
        return "减少背景细节，主体保持不变。", [
            PromptDiff(module="光影", action="replace", new_content="背景减少细节，保持柔和晨光。")
        ], current_prompt + "\n\n反馈修改：远处设施进一步减少。"

    def get_balance(self):
        return {"is_available": True}


class FakeDrawing(OpenAIImageClient):
    requests = []
    fail_partial = False

    def __init__(self, **kwargs):
        kwargs["client"] = SimpleNamespace(images=FakeImages())
        super().__init__(**kwargs)

    def generate_images(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs.copy()))
        if self.fail_partial:
            paths = super().generate_images(prompt, **{**kwargs, "num_images": 1})
            raise ImageGenerationError("模拟第二张失败，已保留第一张。", paths, 2)
        return super().generate_images(prompt, **kwargs)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dummy = self.root / "kb.json"
        atomic_json(dummy, {})
        term = seed_term()
        term["category"] = "art_history"
        term["category_name"] = "艺术流派与视觉美学"
        kb = KnowledgeBase([term], {"categories": [{"id": "art_history", "name": "艺术流派与视觉美学"}]})
        for target, value in [
            ("workflow.PROJECTS", self.root / "projects"), ("storage.WORKSPACE", self.root / "workspace"),
            ("knowledge_base.KB_PATH", dummy), ("api_clients.DeepSeekClient", FakePlanner),
            ("api_clients.OpenAIImageClient", FakeDrawing),
        ]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("knowledge_base.KnowledgeBase.load", return_value=kb)
        patcher.start()
        self.addCleanup(patcher.stop)
        FakePlanner.calls = []
        FakePlanner.fail_integration = False
        FakeDrawing.requests = []
        FakeDrawing.fail_partial = False
        st.cache_resource.clear()
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)

    def run_app(self):
        self.app.run()
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))
        return self.app

    def click(self, label):
        buttons = [button for button in self.app.button if button.label == label]
        self.assertTrue(buttons, label)
        buttons[0].click()
        return self.run_app()

    def area(self, label):
        return next(w for w in self.app.text_area if w.label == label)

    def project(self):
        return self.app.session_state.current_project

    def start(self):
        self.run_app()
        self.area("创作意图").set_value("清晨车站的旅人")
        self.click("提交想法，展开方案")
        self.assertEqual(self.app.selectbox(key="project_selector").value, self.project().project_id)
        self.assertEqual([c[0] for c in FakePlanner.calls], ["analyze"])
        self.assertFalse(FakeDrawing.requests)
        self.assertFalse(self.app.tabs)

    def choose(self, identity="dir_1"):
        next(w for w in self.app.radio if w.label == "选择一个展开方案").set_value(identity)
        self.run_app()

    def develop(self):
        self.start()
        self.choose()
        self.click("提交方案，让 DeepSeek 生成意象清单")
        self.click("确认保留的意象，让 DeepSeek 深化提示词")
        self.assertTrue(self.area("最终提示词（可直接修改）").value)
        self.assertFalse(FakeDrawing.requests)

    def test_unmodified_path_and_repeat_stay_in_same_editor(self):
        self.develop()
        self.assertEqual([c[0] for c in FakePlanner.calls], ["analyze", "imagery", "develop"])
        self.assertEqual([m["id"] for m in FakePlanner.calls[2][2]],
                         ["core_style", "medium_appearance", "viewpoint", "composition",
                          "scene_details", "lighting", "color", "mood", "must_avoid"])
        self.assertEqual(FakePlanner.calls[2][2][0]["knowledge_material"][0]["id"], "sample")
        prompt = self.area("最终提示词（可直接修改）").value
        self.assertLess(prompt.index("【核心风格】"), prompt.index("【场景与细节】"))
        self.click("提交提示词，生成 A/B 图片")
        self.assertIn(prompt, FakeDrawing.requests[-1][0])
        self.assertIn("禁止多余的手臂", FakeDrawing.requests[-1][0])
        self.assertEqual(len(self.project().iterations[-1].outputs), 2)
        self.assertEqual(self.area("最终提示词（可直接修改）").value, prompt)
        self.click("提交提示词，生成 A/B 图片")
        self.assertEqual(len(self.project().iterations), 2)
        self.assertEqual(FakeDrawing.requests[0][0], FakeDrawing.requests[1][0])
        self.assertEqual(len(FakePlanner.calls), 3)  # Repeat drawing never calls DeepSeek.

    def test_module_selection_and_module_knowledge_are_user_controlled(self):
        self.start()
        self.choose()
        modules = next(widget for widget in self.app.multiselect if widget.label == "内容模块（选择 6–8 个）")
        modules.set_value(["core_style", "viewpoint", "composition", "scene_details", "lighting"])
        self.run_app()
        self.assertTrue(next(button for button in self.app.button
                             if button.label == "提交方案，让 DeepSeek 生成意象清单").disabled)
        modules = next(widget for widget in self.app.multiselect if widget.label == "内容模块（选择 6–8 个）")
        modules.set_value(["core_style", "viewpoint", "composition", "scene_details", "lighting", "color"])
        self.run_app()
        core_terms = next(widget for widget in self.app.multiselect if widget.label == "核心风格知识库术语")
        core_terms.set_value([])
        self.area("构图补充要求（可选）").set_value("主体严格位于右侧三分之一，左侧保留留白。")
        self.click("提交方案，让 DeepSeek 生成意象清单")
        self.click("确认保留的意象，让 DeepSeek 深化提示词")
        requests = FakePlanner.calls[2][2]
        self.assertEqual([item["id"] for item in requests],
                         ["core_style", "viewpoint", "composition", "scene_details", "lighting", "color", "must_avoid"])
        self.assertEqual(requests[0]["knowledge_material"], [])
        composition = next(item for item in requests if item["id"] == "composition")
        self.assertIn("右侧三分之一", composition["module_note"])
        self.assertNotIn("【媒介外观】", self.area("最终提示词（可直接修改）").value)

    def test_direction_and_final_edits_reach_apis_and_preserve_history(self):
        self.start()
        self.assertTrue(next(b for b in self.app.button if b.label == "提交方案，让 DeepSeek 生成意象清单").disabled)
        self.choose("dir_2")
        edited_direction = "将方案改成暖色版画：旅人靠左，红色外套，不要蓝色外套。"
        self.area("方案内容（可修改）").set_value(edited_direction)
        self.area("必须保留").set_value("保留红色外套")
        self.click("提交方案，让 DeepSeek 生成意象清单")
        self.click("确认保留的意象，让 DeepSeek 深化提示词")
        self.assertEqual(FakePlanner.calls[1][1]["submitted_direction"], edited_direction)
        self.assertEqual(FakePlanner.calls[1][1]["selected_direction"]["direction_id"], "dir_2")
        self.assertIn("保留红色外套", FakePlanner.calls[2][3])
        self.assertIn(edited_direction, self.area("最终提示词（可直接修改）").value)
        first = "【主体】\n用户手改的红色外套旅人，靠左站立。\n\n【禁止与保持事项】\n不要任何文字。"
        self.area("最终提示词（可直接修改）").set_value(first)
        next(w for w in self.app.selectbox if w.label == "画幅比例").select("16:9")
        self.click("提交提示词，生成 A/B 图片")
        old = self.project().iterations[0].model_dump(mode="json")
        second = first.replace("靠左站立", "靠右坐下") + "\n背景留白更多。"
        self.area("最终提示词（可直接修改）").set_value(second)
        self.run_app()
        self.assertEqual(self.project().iterations[0].model_dump(mode="json"), old)
        self.assertEqual(workflow.load_projects()[0][0].prompt_draft, second)
        self.click("提交提示词，生成 A/B 图片")
        self.assertIn(second, FakeDrawing.requests[-1][0])
        self.assertIn("禁止多余的手臂", FakeDrawing.requests[-1][0])
        self.assertEqual(FakeDrawing.requests[-1][1]["aspect_ratio"], "16:9")
        self.assertEqual(self.project().iterations[0].model_dump(mode="json"), old)
        self.assertIn(second, self.project().iterations[-1].locked_prompt)
        self.assertEqual(self.project().iterations[-1].outputs[0].actual_width, 24)
        self.assertEqual(len(FakePlanner.calls), 3)
        self.click("设为成品 A")
        self.click("保存为案例")
        self.assertTrue(any("已保存" in x.value for x in self.app.success))

    def test_feedback_is_optional_and_adoption_does_not_draw(self):
        self.develop()
        self.click("提交提示词，生成 A/B 图片")
        self.area("最终提示词（可直接修改）").set_value("【主体】\n用户最新手改的蓝外套旅人。\n【光影】\n减少背景细节。")
        self.area("想让 DeepSeek 怎么改").set_value("减少背景")
        self.area("希望保留的点").set_value("保留蓝色外套")
        self.click("让 DeepSeek 提出修改稿")
        self.assertIn("用户最新手改", FakePlanner.calls[-1][1])
        calls = len(FakeDrawing.requests)
        self.click("采用到提示词编辑区")
        self.assertEqual(len(FakeDrawing.requests), calls)
        self.assertIn("反馈修改", self.area("最终提示词（可直接修改）").value)
        lighting = next(module for module in self.project().iterations[-1].prompt_modules if module.title == "光影")
        self.assertEqual(lighting.confirmed_content, "背景减少细节，保持柔和晨光。")
        self.area("最终提示词（可直接修改）").set_value("【主体】\n采用后继续手工修改，只有旅人和大片留白。")
        self.click("提交提示词，生成 A/B 图片")
        self.assertEqual(len(FakeDrawing.requests), calls + 1)
        self.assertIn("采用后继续手工修改", FakeDrawing.requests[-1][0])

    def test_failed_redevelopment_preserves_prompt_and_direction_draft(self):
        self.develop()
        original_prompt = self.area("最终提示词（可直接修改）").value
        self.click("返回方案，重新选择或修改")
        self.choose("dir_2")
        self.area("方案内容（可修改）").set_value("深化失败也必须保留这个修改后的方案。")
        self.click("提交方案，让 DeepSeek 生成意象清单")
        FakePlanner.fail_integration = True
        self.click("确认保留的意象，让 DeepSeek 深化提示词")
        self.assertTrue(self.app.error)
        self.assertEqual(self.project().prompt_draft, original_prompt)
        self.assertEqual(len(self.project().iterations), 1)
        self.assertIn("必须保留", self.project().direction_drafts["dir_2"])
        self.click("返回已有提示词，继续编辑")
        self.assertEqual(self.area("最终提示词（可直接修改）").value, original_prompt)

    def test_reselect_and_redevelop_preserves_edits_and_previous_images(self):
        self.develop()
        self.click("提交提示词，生成 A/B 图片")
        old = self.project().iterations[0].model_dump(mode="json")
        self.click("返回方案，重新选择或修改")
        first_edit = "方案甲的手改版本：旅人在左侧，暖光。"
        self.area("方案内容（可修改）").set_value(first_edit)
        self.run_app()
        self.choose("dir_2")
        second_edit = "方案乙的手改版本：红色外套旅人在右侧，平面版画。"
        self.area("方案内容（可修改）").set_value(second_edit)
        self.run_app()
        self.choose("dir_1")
        self.assertEqual(self.area("方案内容（可修改）").value, first_edit)
        self.choose("dir_2")
        self.assertEqual(self.area("方案内容（可修改）").value, second_edit)
        self.click("提交方案，让 DeepSeek 生成意象清单")
        self.click("确认保留的意象，让 DeepSeek 深化提示词")
        self.assertIn(second_edit, self.area("最终提示词（可直接修改）").value)
        self.assertEqual(len(FakeDrawing.requests), 1)
        self.assertEqual(self.project().selected_direction, "dir_2")
        self.assertEqual(self.project().iterations[0].model_dump(mode="json"), old)
        self.click("提交提示词，生成 A/B 图片")
        self.assertEqual(len(self.project().iterations), 2)
        self.assertIn(second_edit, FakeDrawing.requests[-1][0])
        self.assertEqual(self.project().iterations[0].model_dump(mode="json"), old)

    def test_partial_image_failure_keeps_editor_and_previous_version(self):
        self.develop()
        FakeDrawing.fail_partial = True
        self.click("提交提示词，生成 A/B 图片")
        self.assertEqual(len(self.project().iterations[-1].outputs), 1)
        old = self.project().iterations[-1].model_dump(mode="json")
        self.assertTrue(self.app.error)
        self.area("最终提示词（可直接修改）").set_value("【主体】\n失败后继续编辑，旅人保持清晰，背景变成纯白。")
        FakeDrawing.fail_partial = False
        self.click("提交提示词，生成 A/B 图片")
        self.assertEqual(self.project().iterations[0].model_dump(mode="json"), old)
        self.assertEqual(len(self.project().iterations[-1].outputs), 2)

    def test_empty_editor_disables_drawing_and_stale_feedback_is_cleared(self):
        self.develop()
        self.click("提交提示词，生成 A/B 图片")
        self.area("想让 DeepSeek 怎么改").set_value("减少背景")
        self.click("让 DeepSeek 提出修改稿")
        self.assertIn("feedback", self.app.session_state)
        self.area("最终提示词（可直接修改）").set_value(" ")
        self.run_app()
        self.assertNotIn("feedback", self.app.session_state)
        self.assertTrue(next(b for b in self.app.button if b.label == "提交提示词，生成 A/B 图片").disabled)
        self.assertEqual(workflow.load_projects()[0][0].prompt_draft, " ")

    def test_project_switch_preserves_memory_and_legacy_editor(self):
        for pid in ["one", "two"]:
            project = Project(project_id=pid, display_name=pid, short_name=pid, initial_intent="test")
            if pid == "one":
                workflow.attach_iteration(project, Iteration(revision="r001", iteration_type="initial",
                    locked_prompt="旧项目已有提示词，打开后应该直接允许编辑出图。", prompt_modules=[]))
            workflow.save_project(project)
        self.run_app()
        self.app.selectbox(key="project_selector").select("one")
        self.run_app()
        self.assertIn("旧项目已有提示词", self.area("最终提示词（可直接修改）").value)
        self.project().must_keep = "内存状态不能被旧文件覆盖"
        self.run_app()
        self.assertEqual(self.project().must_keep, "内存状态不能被旧文件覆盖")
        self.area("最终提示词（可直接修改）").set_value("旧项目恢复后手改的提示词，独立保存为当前编辑稿。")
        self.run_app()
        self.app.session_state.feedback = {"scope": "one"}
        self.app.selectbox(key="project_selector").select("two")
        self.run_app()
        self.assertEqual(self.project().project_id, "two")
        self.assertNotIn("feedback", self.app.session_state)
        self.app.selectbox(key="project_selector").select("one")
        self.run_app()
        self.assertIn("独立保存", self.area("最终提示词（可直接修改）").value)

    def test_knowledge_can_be_selected_and_persisted(self):
        workflow.save_project(Project(project_id="one", display_name="one", short_name="one", initial_intent="test"))
        self.run_app()
        self.app.selectbox(key="project_selector").select("one")
        self.run_app()
        self.click("查看术语库")
        self.click("加入当前项目")
        self.assertEqual(self.project().selected_term_ids, ["sample"])
        projects, errors = workflow.load_projects()
        self.assertEqual(projects[0].selected_term_ids, ["sample"])
        self.click("从项目移除")
        self.assertEqual(self.project().selected_term_ids, [])


if __name__ == "__main__":
    unittest.main()
