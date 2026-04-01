"""情节设计学习模块

在场景级文风学习之上，新增 book/chapter 级别的 RL 循环，
学习作者的情节设计思路：
- 节拍分布、角色弧线、伏笔节奏、叙事张力曲线
"""

from .structure import PlotBeat, CharacterArc, Foreshadowing, PlotStructure
from .analyzer import PlotStructureAnalyzer
from .reward import PlotDesignReward, PlotRewardWeights
from .optimizer import PlotDesignOptimizer, PlotPromptState
