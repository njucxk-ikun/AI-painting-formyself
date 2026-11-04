"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ImageRole(str, Enum):
    """参考图用途"""
    CHARACTER = "character"  # 角色参考
    COMPOSITION = "composition"  # 构图参考
    STYLE = "style"  # 风格参考
    LIGHTING = "lighting"  # 光影参考
    COLOR = "color"  # 配色参考


class ReferenceImage(BaseModel):
    """参考图片"""
    ref_id: str  # 稳定ID
    file_path: str
    file_hash: str
    role: ImageRole
    description: str = ""
    upload_time: datetime = Field(default_factory=datetime.now)


class SceneImagery(BaseModel):
    """A concrete visible element proposed by DeepSeek and reviewed by the user."""
    imagery_id: str
    category: Literal["subject", "character", "environment", "object", "action", "visual_symbol", "effect"]
    name: str = Field(min_length=1, max_length=24)
    # Kept for backward compatibility; prompt writing uses quantity_description instead.
    exact_count: int | None = Field(default=None, ge=1, le=50)
    quantity_description: str = Field(default="", min_length=0, max_length=80)
    quantity_mode: Literal["exact", "aggregate", "continuous", "distributed"] = "exact"
    position: str = Field(min_length=2, max_length=48)
    appearance: str = Field(min_length=2, max_length=72)
    state_action: str = Field(min_length=2, max_length=48)
    relationship: str = Field(min_length=2, max_length=64)
    importance: Literal["primary", "secondary", "supporting"] = "secondary"
    selected: bool = True
    optional_candidate: bool = False
    reason: str = Field(default="", max_length=60)


class StyleCandidate(BaseModel):
    """An optional, user-selectable style reference suggested for a direction."""
    candidate_id: str
    name: str
    english_name: str = ""
    term_id: str | None = None
    prompt_phrase: str
    rationale: str = ""
    selected: bool = False


class DynamicModule(BaseModel):
    """动态模块"""
    module_id: str
    title: str
    order: int
    recommended_direction: Optional[str] = None
    recommend_reason: Optional[str] = None
    user_input: str = ""
    ai_draft: str = ""
    confirmed_content: str = ""
    knowledge_term_ids: List[str] = Field(default_factory=list)
    imagery_ids: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)


class PromptDiff(BaseModel):
    """提示词差异"""
    module: str
    action: Literal["add", "remove", "replace"]
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    reason: str = ""


class GenerationOutput(BaseModel):
    """生成输出"""
    image_path: str
    slot: Literal["A", "B"]  # A图或B图
    generated_at: datetime = Field(default_factory=datetime.now)
    selected_as_reference: bool = False
    selected_as_final: bool = False
    user_rating: Optional[int] = None
    user_notes: str = ""
    actual_width: Optional[int] = None
    actual_height: Optional[int] = None


class Iteration(BaseModel):
    """迭代版本"""
    revision: str  # r001, r002...
    iteration_type: Literal["initial", "redraw", "feedback"]  # 初始/重抽/反馈修改
    source_image: Optional[str] = None  # 基于哪张图
    user_feedback: str = ""
    keep_aspects: str = ""  # 需要保留的点
    change_requests: str = ""  # 需要改变的点
    
    # 提示词版本
    locked_prompt: str
    prompt_modules: List[DynamicModule]
    prompt_diff: List[PromptDiff] = []
    
    # 生成参数
    aspect_ratio: str = "1:1"
    output_size: str = "1024x1024"
    model_name: str = ""
    
    # 参考图
    reference_images: List[ReferenceImage] = []
    
    # 生成结果
    outputs: List[GenerationOutput] = []
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    api_calls: int = 0  # API调用次数
    success: bool = False
    error_message: str = ""
    interaction_model: str = ""
    knowledge_version: str = ""
    knowledge_term_ids: List[str] = Field(default_factory=list)
    needs_integration: bool = False
    direction_id: str = ""
    direction_text: str = ""
    constraints_snapshot: str = ""
    prompt_module_ids: List[str] = Field(default_factory=list)
    module_term_ids: Dict[str, List[str]] = Field(default_factory=dict)
    module_notes: Dict[str, str] = Field(default_factory=dict)
    planning_snapshot: Dict[str, Any] = Field(default_factory=dict)
    confirmed_imagery: List[SceneImagery] = Field(default_factory=list)
    detail_mode: Literal["clean", "balanced", "rich"] = "balanced"
    contains_people: bool = False


class VisualDirection(BaseModel):
    """AI推荐的视觉方向"""
    direction_id: str
    name: str
    technical_terms: List[str]
    explanation: str
    visual_consequences: str
    applicable_reason: str
    is_default: bool = False
    expanded_idea: str = ""  # 可供用户直接修改的完整创意展开，兼容旧项目
    recommended_module_ids: List[str] = Field(default_factory=list)
    knowledge_term_ids: List[str] = Field(default_factory=list)
    contains_people: bool = False
    style_candidates: List[StyleCandidate] = Field(default_factory=list, max_length=10)


class IntentAnalysis(BaseModel):
    """意图分析结果"""
    restated_goal: str  # 准确复述
    main_challenges: List[str]  # 视觉矛盾或难点
    visual_directions: List[VisualDirection]
    suggested_modules: List[str] = Field(default_factory=list)  # 旧项目兼容；新项目按方向保存固定模块ID


class Project(BaseModel):
    """项目"""
    project_id: str
    short_name: str
    display_name: str
    
    # 创作意图
    initial_intent: str
    mode: Literal["text2img", "img2img"] = "text2img"
    
    # AI分析
    intent_analysis: Optional[IntentAnalysis] = None
    selected_direction: Optional[str] = None
    selected_direction_text: str = ""
    direction_drafts: Dict[str, str] = Field(default_factory=dict)
    direction_module_ids: Dict[str, List[str]] = Field(default_factory=dict)
    direction_module_term_ids: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    direction_module_notes: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    direction_constraint_modules: Dict[str, List[str]] = Field(default_factory=dict)
    direction_imagery: Dict[str, List[SceneImagery]] = Field(default_factory=dict)
    direction_imagery_source_text: Dict[str, str] = Field(default_factory=dict)
    direction_imagery_detail_modes: Dict[str, Literal["clean", "balanced", "rich"]] = Field(default_factory=dict)
    direction_detail_modes: Dict[str, Literal["clean", "balanced", "rich"]] = Field(default_factory=dict)
    direction_style_candidate_ids: Dict[str, List[str]] = Field(default_factory=dict)
    prompt_draft: Optional[str] = None  # 当前可编辑稿；已出图版本另存不可变快照
    prompt_draft_version: int = 0
    draft_aspect_ratio: str = ""
    
    # 固定约束
    must_keep: str = ""
    must_avoid: str = ""
    locked_constraints: str = ""
    selected_term_ids: List[str] = Field(default_factory=list)
    reference_images: List[ReferenceImage] = Field(default_factory=list)
    
    # 迭代历史
    iterations: List[Iteration] = []
    current_revision: str = "r000"
    
    # 成品
    final_images: List[str] = []
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: Literal["draft", "in_progress", "completed"] = "draft"
    tags: List[str] = []


class CaseLibraryEntry(BaseModel):
    """案例库条目"""
    case_id: str
    project_id: str
    title: str
    
    # 内容快照
    initial_intent: str
    used_style_terms: List[str]
    final_modules: List[DynamicModule]
    final_prompt: str
    final_images: List[str]
    
    # 经验总结
    iterations_count: int
    key_learnings: str = ""
    user_comments: str = ""
    
    # 元数据
    created_at: datetime
    tags: List[str] = []


class StyleTerm(BaseModel):
    """风格术语"""
    id: str
    name: str
    aliases: List[str]
    category: str
    nature: str  # 历史流派/技术手法/通俗描述/个人自定义标签
    description: str
    visual_effect: str
    applicable_scenes: List[str]
    prompt_keywords: List[str]
    source: str
    examples: List[str] = []
    user_notes: str = ""
    status: Literal["active", "inactive"] = "active"
