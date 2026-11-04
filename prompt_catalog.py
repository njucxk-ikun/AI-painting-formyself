"""Fixed prompt-module catalog and knowledge routing for image planning."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptModuleSpec:
    module_id: str
    title: str
    purpose: str
    knowledge_categories: tuple[str, ...] = ()
    search_hint: str = ""
    conditional: bool = False


MODULE_SPECS = (
    PromptModuleSpec(
        "core_style", "核心风格",
        "确定画面的主要视觉语言，只选择彼此兼容、真正影响整体观感的风格。",
        ("art_history", "anime", "graphic", "culture", "narrative"),
        "艺术风格 视觉美学 插画 摄影 抽象 写实",
    ),
    PromptModuleSpec(
        "medium_appearance", "媒介外观",
        "说明图像像什么媒介或制作方式，包括笔触、颗粒、材质、渲染与表面质感。",
        ("method", "line_texture", "photo", "anime", "graphic"),
        "媒介 材质 笔触 纸张 颜料 胶片 3D 渲染 纹理",
    ),
    PromptModuleSpec(
        "viewpoint", "视角",
        "说明景别、机位、观察方向、透视和主体在镜头中的尺度。",
        ("camera", "photo"),
        "景别 机位 视角 镜头 透视 焦距",
    ),
    PromptModuleSpec(
        "composition", "构图",
        "规定主体位置、画面重心、留白、层次、引导线和元素之间的空间关系。",
        ("composition", "graphic", "camera", "narrative"),
        "构图 留白 重心 层次 引导线 视觉组织",
    ),
    PromptModuleSpec(
        "scene_details", "场景与细节",
        "按当前画面动态展开环境背景、主体角色、动作神态、关键物件等必要子项。",
        ("environment", "character_design", "pose", "emotion", "ornament", "narrative"),
        "环境 场景 主体 角色 动作 神态 道具 物件 空间叙事",
    ),
    PromptModuleSpec(
        "lighting", "光影",
        "说明光源方向、软硬、明暗关系、阴影和空气效果。",
        ("lighting", "photo"),
        "光线 光源 阴影 明暗 大气 曝光",
    ),
    PromptModuleSpec(
        "color", "色彩",
        "说明主色、辅助色、冷暖、饱和度、明度和配色关系。",
        ("composition", "line_texture", "photo", "art_history", "anime", "graphic"),
        "色彩 配色 色调 冷暖 饱和度 明度 调色板",
    ),
    PromptModuleSpec(
        "mood", "意境",
        "概括画面需要传达的情绪和叙事余味，并落到能看见的氛围线索。",
        ("emotion", "narrative", "environment", "art_history"),
        "情绪 氛围 意境 叙事感 宁静 紧张 梦境",
    ),
    PromptModuleSpec(
        "must_keep", "绝对需要遵守的内容",
        "只记录身份、位置、数量、文字、参考图一致性等不可妥协要求。",
        conditional=True,
    ),
    PromptModuleSpec(
        "must_avoid", "绝对需要避免的内容",
        "仅在禁用项较多或容易误生成时单列；否则合并进绝对需要遵守的内容。",
        conditional=True,
    ),
)

MODULE_BY_ID = {spec.module_id: spec for spec in MODULE_SPECS}
CONTENT_MODULE_IDS = tuple(spec.module_id for spec in MODULE_SPECS if not spec.conditional)
CONSTRAINT_MODULE_IDS = tuple(spec.module_id for spec in MODULE_SPECS if spec.conditional)
DEFAULT_CONTENT_MODULE_IDS = CONTENT_MODULE_IDS
MIN_CONTENT_MODULES = 6
MAX_CONTENT_MODULES = 8

ANATOMY_CONSTRAINT = (
    "人物肢体数量与人体结构必须正确；禁止多余的手臂、手、腿、脚、手指、"
    "重复肢体、融合肢体、断裂肢体或错误关节。"
)

DETAIL_MODE_LABELS = {
    "clean": "干净克制",
    "balanced": "均衡",
    "rich": "丰富细节",
}

DETAIL_MODE_POLICIES = {
    "clean": (
        "干净克制：主体优先，轮廓清楚，背景简化，只保留少量必要道具；"
        "二次元内容优先清爽线稿、平涂或克制的赛璐璐阴影。除非用户明确要求，"
        "不要添加装饰粒子、泛光、轮廓光、散景、复杂纹理、华丽光效、电影化调色、"
        "密集环境叙事或铺满画面的细节。"
    ),
    "balanced": (
        "均衡：主体、环境和光色各保留足以说明画面的信息，背景与装饰不得抢过主体；"
        "不自动添加粒子、泛光、复杂笔触或电影化效果。"
    ),
    "rich": (
        "丰富细节：可以增加与主题直接相关的环境、材质和光影层次，但仍需明确主次，"
        "不得以无关装饰、重复纹理或同义形容词堆满画面。"
    ),
}


def detail_mode_policy(mode):
    return DETAIL_MODE_POLICIES.get(mode, DETAIL_MODE_POLICIES["balanced"])

LEGACY_TITLE_MAP = {
    "核心风格": "core_style",
    "概念与风格": "core_style",
    "媒介外观": "medium_appearance",
    "视角": "viewpoint",
    "构图": "composition",
    "构图与视角": "composition",
    "场景与细节": "scene_details",
    "主体与动作": "scene_details",
    "主体角色与动作": "scene_details",
    "光影": "lighting",
    "光影与氛围": "lighting",
    "色彩": "color",
    "意境": "mood",
    "禁止与保持事项": "must_keep",
    "绝对需要遵守的内容": "must_keep",
    "绝对需要避免的内容": "must_avoid",
}


def normalize_module_ids(values, *, content_only=False):
    """Normalize model/legacy values while preserving fixed catalog order."""
    requested = set()
    for value in values or []:
        identity = value if value in MODULE_BY_ID else LEGACY_TITLE_MAP.get(str(value).strip())
        if identity and (not content_only or identity in CONTENT_MODULE_IDS):
            requested.add(identity)
    source = CONTENT_MODULE_IDS if content_only else tuple(MODULE_BY_ID)
    return [identity for identity in source if identity in requested]


def valid_content_selection(values):
    normalized = normalize_module_ids(values, content_only=True)
    return (MIN_CONTENT_MODULES <= len(normalized) <= MAX_CONTENT_MODULES
            and "scene_details" in normalized)


def recommended_content_modules(values):
    """Return a usable 6-8 module selection for legacy or malformed responses."""
    normalized = normalize_module_ids(values, content_only=True)
    if "scene_details" not in normalized:
        normalized.append("scene_details")
    for identity in DEFAULT_CONTENT_MODULE_IDS:
        if len(normalized) >= MIN_CONTENT_MODULES:
            break
        if identity not in normalized:
            normalized.append(identity)
    chosen = set(normalized[:MAX_CONTENT_MODULES])
    return [identity for identity in CONTENT_MODULE_IDS if identity in chosen]


def title_for(module_id):
    return MODULE_BY_ID[module_id].title


def prompt_module_contract():
    return [
        {"id": spec.module_id, "title": spec.title, "purpose": spec.purpose,
         "conditional": spec.conditional}
        for spec in MODULE_SPECS
    ]


REFERENCE_WRITING_GUIDE = """通用绘画提示词范式：
- 核心风格说明整体视觉语言；媒介外观单独说明笔触、颗粒、材质或渲染质感，不把二者混为同义词堆叠。
- 视角负责景别、机位和透视；构图负责主体位置、留白、层次和视觉引导，两者必须相互兼容。
- 场景与细节按当前图片动态使用“环境背景、主体角色、动作与神态、关键物件”等子项，只写实际需要的子项。
- 光影、色彩、意境分别负责照明关系、配色关系和整体情绪，避免重复描述同一句氛围。
- 绝对约束只保留真正不可妥协的身份、数量、位置、文字或参考一致性要求；普通偏好不要伪装成绝对命令。
- 可以在确有帮助时使用8K、最佳品质等质量词，但它们不是必填项，也不能替代主体、构图、光影和材质的具体描述。
- 二次元和轻小说插画默认追求清楚的角色轮廓、简洁线稿、有限阴影层级和可呼吸的背景；不要把“高质量”误写成粒子、泛光、华丽纹理、电影调色和无处不在的细节。
- 已确认的画面意象构成元素白名单：每个保留意象写清数量、位置、外观、状态/动作和与其他元素的关系；被取消的意象不得在后续重新加入。
- 此范式来自用户参考文档的结构提炼，禁止复用其中的具体角色、颜色、场景、动作、品牌或专有风格组合。"""
