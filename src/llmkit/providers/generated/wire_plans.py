# Code generated — DO NOT EDIT.


from __future__ import annotations

from dataclasses import dataclass, field

#
#
#
#
#
#
#


@dataclass(frozen=True)
class FieldBinding:
    path: str
    source: str
    option_key: str = ""
    const_json: str = ""
    default_json: str = ""
    transform: str = "None"
    omit_if_empty: bool = False


@dataclass(frozen=True)
class BodyPlan:
    label: str
    bindings: tuple[FieldBinding, ...] = field(default_factory=tuple)


PLAN_VIDEO_BEDROCK = BodyPlan(
    label="video-bedrock",
    bindings=(
        FieldBinding(path="modelId", source="Model", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="modelInput.taskType", source="Const", option_key="", const_json="\"TEXT_VIDEO\"", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="modelInput.textToVideoParams.text", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="outputDataConfig.s3OutputDataConfig.s3Uri", source="Option", option_key="output_uri", const_json="", default_json="", transform="None", omit_if_empty=False),
    ),
)

PLAN_VIDEO_GROK = BodyPlan(
    label="video-grok",
    bindings=(
        FieldBinding(path="model", source="Model", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="prompt", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="image.url", source="MediaRef", option_key="", const_json="", default_json="", transform="DataUri", omit_if_empty=True),
    ),
)

PLAN_VIDEO_MODEL_PROMPT = BodyPlan(
    label="video-model-prompt",
    bindings=(
        FieldBinding(path="model", source="Model", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="prompt", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
    ),
)

PLAN_VIDEO_PIX_VERSE = BodyPlan(
    label="video-pixverse",
    bindings=(
        FieldBinding(path="model", source="Model", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="prompt", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="duration", source="Option", option_key="duration", const_json="", default_json="5", transform="None", omit_if_empty=False),
        FieldBinding(path="quality", source="Option", option_key="quality", const_json="", default_json="\"540p\"", transform="None", omit_if_empty=False),
        FieldBinding(path="aspect_ratio", source="Option", option_key="aspect_ratio", const_json="", default_json="\"16:9\"", transform="None", omit_if_empty=False),
    ),
)

PLAN_VIDEO_QWEN = BodyPlan(
    label="video-qwen",
    bindings=(
        FieldBinding(path="input.prompt", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
        FieldBinding(path="model", source="Model", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
    ),
)

PLAN_VIDEO_VEO_INSTANCES = BodyPlan(
    label="video-veo-instances",
    bindings=(
        FieldBinding(path="instances[0].prompt", source="Prompt", option_key="", const_json="", default_json="", transform="None", omit_if_empty=False),
    ),
)


#
VIDEO_BODY_PLANS: dict[str, BodyPlan] = {
    "VideoBedrock": PLAN_VIDEO_BEDROCK,
    "VideoGrok": PLAN_VIDEO_GROK,
    "VideoMinimax": PLAN_VIDEO_MODEL_PROMPT,
    "VideoPixVerse": PLAN_VIDEO_PIX_VERSE,
    "VideoQwen": PLAN_VIDEO_QWEN,
    "VideoTogether": PLAN_VIDEO_MODEL_PROMPT,
    "VideoVeo": PLAN_VIDEO_VEO_INSTANCES,
    "VideoVertexVeo": PLAN_VIDEO_VEO_INSTANCES,
    "VideoVidu": PLAN_VIDEO_MODEL_PROMPT,
    "VideoZhipu": PLAN_VIDEO_MODEL_PROMPT,
}
