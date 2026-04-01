"""情节设计数据结构"""

from dataclasses import dataclass, field


@dataclass
class PlotBeat:
    """单个情节节拍"""
    index: int
    beat_type: str        # setup / complication / climax / resolution / twist / revelation
    summary: str
    tension_level: float  # 0.0 ~ 1.0
    characters: list[str] = field(default_factory=list)
    emotional_tone: str = ""


@dataclass
class CharacterArc:
    """角色弧线节点"""
    character: str
    scene_index: int
    motivation: str
    state: str
    relationship_changes: list[str] = field(default_factory=list)


@dataclass
class Foreshadowing:
    """伏笔记录"""
    setup_scene: int
    payoff_scene: int      # -1 = 尚未回收
    element: str
    subtlety: float = 0.5


@dataclass
class PlotStructure:
    """完整的情节结构"""
    beats: list[PlotBeat] = field(default_factory=list)
    character_arcs: list[CharacterArc] = field(default_factory=list)
    foreshadowings: list[Foreshadowing] = field(default_factory=list)
    tension_curve: list[float] = field(default_factory=list)
    pacing_profile: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "beats": [
                {"index": b.index, "type": b.beat_type, "summary": b.summary,
                 "tension": b.tension_level, "characters": b.characters, "tone": b.emotional_tone}
                for b in self.beats
            ],
            "character_arcs": [
                {"character": a.character, "scene": a.scene_index,
                 "motivation": a.motivation, "state": a.state,
                 "relationship_changes": a.relationship_changes}
                for a in self.character_arcs
            ],
            "foreshadowings": [
                {"setup": f.setup_scene, "payoff": f.payoff_scene,
                 "element": f.element, "subtlety": f.subtlety}
                for f in self.foreshadowings
            ],
            "tension_curve": self.tension_curve,
            "pacing_profile": self.pacing_profile,
        }
