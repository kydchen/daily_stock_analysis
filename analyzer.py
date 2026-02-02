# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',
}


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版
    
    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str
    
    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    confidence_level: str = "中"  # 置信度：高/中/低
    
    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据
    
    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）
    
    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析
    
    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点
    
    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题
    
    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由
    
    # ========== 元数据 ==========
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
        }
    
    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary
    
    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice
    
    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}
    
    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []
    
    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []
    
    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器
    
    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果
    
    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """
    
    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================
    
    SYSTEM_PROMPT = """你是一位顶尖的全球资产投资分析师，擅长利用宏观环境、技术指标和基本面进行跨资产（美股、BTC、黄金、A股）深度分析。

## 👤 用户画像：长期主义者 (Long-term Value & Trend)
- **核心哲学**：注重资产的长期基本面和 200 日线（MA200）以上的长线趋势。
- **择时逻辑**：平时少操作，仅在发生“短期大幅波动”（即：偏离长线均线过远的极致乖离，或触及历史性强支撑/压力位）时才进行择时。
- **交易战术**：偏好分批入场（探针仓），不参与短线博弈。

## 🚨 分析规则与致命弱点优先 (Fatal Weakness First)
1. **第一句话原则**：`analysis_summary` 的**第一句话**必须直击痛点，指出该标的当前最致命的弱点或风险。
2. **长线过滤器**：如果标的价格在 MA200 之下，无论短线如何反弹，操作建议必须保持谨慎。
3. **宏观定价锚点**：
    - 分析 **BTC/黄金**：必须对比 **DXY (美元指数)** 的强弱。
    - 分析 **美股科技股 (如 NVDA)**：必须对比 **US10Y (10年期美债收益率)** 的走势。
    - 如果宏观环境不支持（如加息预期升温），即使技术面多头，也要调低 `sentiment_score`。

## 📈 技术面与实时状态 (Real-time Context)
- **盘前/盘后敏感**：若当前处于 Pre-market (盘前)，必须结合盘前跳空情况给出即时判断。
- **乖离率极致择时**：
    - 乖离率 (Bias MA5) > 5% 或超出 ATR 波动区间：定义为“极致超买”，提示择时减仓或停止买入。
    - 乖离率 < -10% 且处于长期上涨趋势：定义为“极致超卖/黄金坑”，提示择时分批入场。

## 核心交易理念（必须严格遵守）

### 1. 严进策略（不追高）
- **绝对不追高**：当股价偏离 MA5 超过 5% 时，坚决不买入
- **乖离率公式**：(现价 - MA5) / MA5 × 100%
- 乖离率 < 2%：最佳买点区间
- 乖离率 2-5%：可小仓介入
- 乖离率 > 5%：严禁追高！直接判定为"观望"

### 2. 趋势交易（顺势而为）
- **多头排列必须条件**：MA5 > MA10 > MA20，且是基本要求。
- **生命线**：MA200 是牛熊分界。长期主义者只在 MA200 走平或向上的标的中寻找机会。
- 只做多头排列的股票，空头排列坚决不碰
- 均线发散上行优于均线粘合
- 趋势强度判断：看均线间距是否在扩大

### 3. 效率优先（筹码结构）
- 关注筹码集中度：90%集中度 < 15% 表示筹码集中
- 获利比例分析：70-90% 获利盘时需警惕获利回吐
- 平均成本与现价关系：现价高于平均成本 5-15% 为健康

### 4. 买点偏好（回踩支撑）
- **最佳买点**：缩量回踩 MA5 获得支撑
- **次优买点**：回踩 MA10 获得支撑
- **观望情况**：跌破 MA20 时观望

### 5. 风险排查重点
- 减持公告（股东、高管减持）
- 业绩预亏/大幅下滑
- 监管处罚/立案调查
- 行业政策利空
- 大额解禁

## 🚨 致命弱点优先规则 (Fatal Weakness First)
1. 在任何分析生成前，必须首先识别一个**足以推论该标的走势彻底反转或失败**的致命因素。
2. 必须在 `analysis_summary` 的第一句话直接指出该弱点。
3. 如果当前处于宏观逆风（如 DXY 上涨而你正在分析黄金），必须将其列为第一风险。

## 输出格式：决策仪表盘 JSON

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：

```json
{
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "confidence_level": "高/中/低",
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "【致命弱点置顶】+ 一句话核心结论（30字以内，直接告诉用户做什么）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "长期配置/短期择时/立即行动",
            "market_state_note": "当前市场状态(盘前/盘后/交易中)及价格反馈",
            "position_advice": {
                "no_position": "空仓者建议：具体操作指引",
                "has_position": "持仓者建议：具体操作指引"
            }
        },
        "macro_linkage": {
            "dxy_correlation": "美元指数对本资产的影响解读",
            "us10y_correlation": "美债收益率对本资产的影响解读",
            "macro_sentiment": "偏多/中性/偏空"
        },
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线排列状态描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值,
                "ma5": MA5数值,
                "ma10": MA10数值,
                "ma20": MA20数值,
                "bias_ma5": 乖离率百分比数值,
                "bias_status": "安全/警戒/危险",
                "support_level": 支撑位价格,
                "resistance_level": 压力位价格
            },
            "volume_analysis": {
                "volume_ratio": 量比数值,
                "volume_status": "放量/缩量/平量",
                "turnover_rate": 换手率百分比,
                "volume_meaning": "量能含义解读（如：缩量回调表示抛压减轻）"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "avg_cost": 平均成本,
                "concentration": 筹码集中度,
                "chip_health": "健康/一般/警惕"
            }
        },
        
        "intelligence": {
            "latest_news": "【最新消息】近期重要新闻摘要",
            "risk_alerts": ["风险点1：具体描述", "风险点2：具体描述"],
            "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
            "earnings_outlook": "业绩预期分析（基于年报预告、业绩快报等）",
            "sentiment_summary": "舆情情绪一句话总结"
        },
        
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想买入点：XX元（在MA5附近）",
                "secondary_buy": "次优买入点：XX元（在MA10附近）",
                "stop_loss": "止损位：XX元（跌破MA20或X%）",
                "take_profit": "目标位：XX元（前高/整数关口）"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "分批建仓策略描述",
                "risk_control": "风控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 检查项1：多头排列",
                "✅/⚠️/❌ 检查项2：乖离率<5%",
                "✅/⚠️/❌ 检查项3：量能配合",
                "✅/⚠️/❌ 检查项4：无重大利空",
                "✅/⚠️/❌ 检查项5：筹码健康"
            ]
        }
    },
    
    "analysis_summary": "首句必须是致命弱点。随后跟100字综合分析。",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用交易理念",
    
    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点",
    
    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```

## 评分标准

### 强烈买入（80-100分）：
- ✅ 多头排列：MA5 > MA10 > MA20
- ✅ 低乖离率：<2%，最佳买点
- ✅ 缩量回调或放量突破
- ✅ 筹码集中健康
- ✅ 消息面有利好催化

### 买入（60-79分）：
- ✅ 多头排列或弱势多头
- ✅ 乖离率 <5%
- ✅ 量能正常
- ⚪ 允许一项次要条件不满足

### 观望（40-59分）：
- ⚠️ 乖离率 >5%（追高风险）
- ⚠️ 均线缠绕趋势不明
- ⚠️ 有风险事件

### 卖出/减仓（0-39分）：
- ❌ 空头排列
- ❌ 跌破MA20
- ❌ 放量下跌
- ❌ 重大利空

## 决策仪表盘核心原则

1. **核心结论先行**：一句话说清该买该卖
2. **分持仓建议**：空仓者和持仓者给不同建议
3. **精确狙击点**：必须给出具体价格，不说模糊的话
4. **检查清单可视化**：用 ✅⚠️❌ 明确显示每项检查结果
5. **风险优先级**：舆情中的风险点要醒目标出"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化 AI 分析器
        
        优先级：Gemini > OpenAI 兼容 API
        
        Args:
            api_key: Gemini API Key（可选，默认从配置读取）
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._target_model = model_name  # 🆕 保存指定模型
        self._model = None
        self._current_model_name = None  # 当前使用的模型名称
        self._using_fallback = False  # 是否正在使用备选模型
        self._use_openai = False  # 是否使用 OpenAI 兼容 API
        self._openai_client = None  # OpenAI 客户端
        
        # 检查 Gemini API Key 是否有效（过滤占位符）
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10
        
        # 优先尝试初始化 Gemini
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}，尝试 OpenAI 兼容 API")
                self._init_openai_fallback()
        else:
            # Gemini Key 未配置，尝试 OpenAI
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()
        
        # 两者都未配置
        if not self._model and not self._openai_client:
            logger.warning("未配置任何 AI API Key，AI 分析功能将不可用")
    
    def _init_openai_fallback(self) -> None:
        """
        初始化 OpenAI 兼容 API 作为备选
        
        支持所有 OpenAI 格式的 API，包括：
        - OpenAI 官方
        - DeepSeek
        - 通义千问
        - Moonshot 等
        """
        config = get_config()
        
        # 检查 OpenAI API Key 是否有效（过滤占位符）
        openai_key_valid = (
            config.openai_api_key and 
            not config.openai_api_key.startswith('your_') and 
            len(config.openai_api_key) > 10
        )
        
        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置或配置无效")
            return
        
        # 分离 import 和客户端创建，以便提供更准确的错误信息
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库，请运行: pip install openai")
            return
        
        try:
            # base_url 可选，不填则使用 OpenAI 官方默认地址
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url
            
            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功 (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # 依赖缺失（如 socksio）
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI 客户端需要 SOCKS 代理支持，请运行: pip install httpx[socks] 或 pip install socksio")
            else:
                logger.error(f"OpenAI 依赖缺失: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI 代理配置错误: {e}，如使用 SOCKS 代理请运行: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI 兼容 API 初始化失败: {e}")
    
    def _init_model(self) -> None:
        """
        初始化 Gemini 模型
        
        配置：
        - 使用 gemini-3-flash-preview 或 gemini-2.5-flash 模型
        - 不启用 Google Search（使用外部 Tavily/SerpAPI 搜索）
        """
        try:
            import google.generativeai as genai
            
            # 配置 API Key
            genai.configure(api_key=self._api_key)
            
            # 从配置获取模型名称
            config = get_config()
            # model_name = config.gemini_model
            # fallback_model = config.gemini_model_fallback
            # 🆕 修改：优先使用初始化时指定的模型，否则用默认配置
            model_name = self._target_model or config.gemini_model
            fallback_model = config.gemini_model_fallback
            
            # 不再使用 Google Search Grounding（已知有兼容性问题）
            # 改为使用外部搜索服务（Tavily/SerpAPI）预先获取新闻
            
            # 尝试初始化主模型
            try:
                self._model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = model_name
                self._using_fallback = False
                logger.info(f"Gemini 模型初始化成功 (模型: {model_name})")
            except Exception as model_error:
                # 尝试备选模型
                logger.warning(f"主模型 {model_name} 初始化失败: {model_error}，尝试备选模型 {fallback_model}")
                self._model = genai.GenerativeModel(
                    model_name=fallback_model,
                    system_instruction=self.SYSTEM_PROMPT,
                )
                self._current_model_name = fallback_model
                self._using_fallback = True
                logger.info(f"Gemini 备选模型初始化成功 (模型: {fallback_model})")
            
        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._model = None
    
    def _switch_to_fallback_model(self) -> bool:
        """
        切换到备选模型
        
        Returns:
            是否成功切换
        """
        try:
            import google.generativeai as genai
            config = get_config()
            fallback_model = config.gemini_model_fallback
            
            logger.warning(f"[LLM] 切换到备选模型: {fallback_model}")
            self._model = genai.GenerativeModel(
                model_name=fallback_model,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] 备选模型 {fallback_model} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"[LLM] 切换备选模型失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self._model is not None or self._openai_client is not None
    
    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        调用 OpenAI 兼容 API
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._openai_client.chat.completions.create(
                    model=self._current_model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=generation_config.get('temperature', 0.7),
                    max_tokens=generation_config.get('max_output_tokens', 8192),
                )
                
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API 返回空响应")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API 调用失败，已达最大重试次数")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        调用 AI API，带有重试和模型切换机制
        
        优先级：Gemini > Gemini 备选模型 > OpenAI 兼容 API
        
        处理 429 限流错误：
        1. 先指数退避重试
        2. 多次失败后切换到备选模型
        3. Gemini 完全失败后尝试 OpenAI
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        # 如果已经在使用 OpenAI 模式，直接调用 OpenAI
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # 请求前增加延时（防止请求过快触发限流）
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                    delay = min(delay, 60)  # 最大60秒
                    logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 120}
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini 返回空响应")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是 429 限流错误
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[Gemini] API 限流 (429)，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                    
                    # 如果已经重试了一半次数且还没切换过备选模型，尝试切换
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] 已切换到备选模型，继续重试")
                        else:
                            logger.warning("[Gemini] 切换备选模型失败，继续使用当前模型重试")
                else:
                    # 非限流错误，记录并继续重试
                    logger.warning(f"[Gemini] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
        
        # Gemini 所有重试都失败，尝试 OpenAI 兼容 API
        if self._openai_client:
            logger.warning("[Gemini] 所有重试失败，切换到 OpenAI 兼容 API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # 尝试懒加载初始化 OpenAI
            logger.warning("[Gemini] 所有重试失败，尝试初始化 OpenAI 兼容 API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                    raise last_error or openai_error
        
        # 所有方式都失败
        raise last_error or Exception("所有 AI API 调用失败，已达最大重试次数")
        
    def _determine_asset_type(self, code: str) -> str:
        """简单的资产类型判断逻辑"""
        if code.endswith('=F') or code.endswith('=X') or code in ['GLD', 'SLV']:
            return "COMMODITY"  # 商品/黄金
        elif '-' in code and ('USD' in code or 'USDT' in code):
            return "CRYPTO"     # 加密货币
        elif code.isdigit() and len(code) == 6:
            return "ASHARE"     # A股
        else:
            return "US_STOCK"   # 美股 (默认)


    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票（加固版：解决数据断链与 NoneType 崩溃问题）
        """
        # 1. 基础上下文拦截
        if not context:
            logger.error("分析上下文 context 为空")
            return self._return_error_result("Unknown", "Empty context")

        code = context.get('code', 'Unknown')
        config = get_config()

        # 2.【核心加固】强制字典化：解决 'NoneType' object has no attribute 'get'
        # 即使数据源抓取失败导致对应字段为 None，这里也会强制变为空字典，确保后续 .get() 安全
        realtime = context.get('realtime') if context.get('realtime') is not None else {}
        today = context.get('today') if context.get('today') is not None else {}
        trend = context.get('trend_analysis') if context.get('trend_analysis') is not None else {}

        # 3. 智能获取股票名称（多级降级，解决日志中 Asset-Unknown 的问题）
        # 优先级：上下文显式指定 > 实时行情返回 > 映射表 > 默认占位符
        name = (
            context.get('stock_name') or 
            realtime.get('name') or 
            STOCK_NAME_MAP.get(code) or 
            f"Asset-{code}"
        )

        # 4. 检查模型可用性
        if not self.is_available():
            return AnalysisResult(
                code=code, name=name, sentiment_score=50, trend_prediction='震荡',
                operation_advice='持有', confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 AI API Key 后重试', success=False,
                error_message='AI API Key 未配置'
            )

        # 5. 请求前流控延时
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)

        try:
            # 6. 重新将处理过的安全字典塞回 context 供 _format_prompt 使用
            safe_context = context.copy()
            safe_context['realtime'] = realtime
            safe_context['today'] = today
            safe_context['trend_analysis'] = trend

            # 7. 格式化输入
            prompt = self._format_prompt(safe_context, name, news_context)
            
            # 获取模型信息用于日志
            model_info = self._current_model_name or "Active Model"
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_info} | 新闻: {'是' if news_context else '否'}")
            
            # 8. 设置生成配置并调用 API (带重试)
            generation_config = {"temperature": 0.7, "max_output_tokens": 8192}
            start_time = time.time()
            response_text = self._call_api_with_retry(prompt, generation_config)
            elapsed = time.time() - start_time
            
            logger.info(f"[LLM返回] 响应成功, 耗时 {elapsed:.2f}s")
            
            # 9. 解析响应
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            
            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {str(e)}")
            return AnalysisResult(
                code=code, name=name, sentiment_score=50, trend_prediction='异常',
                operation_advice='持有', confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试', success=False,
                error_message=str(e)
            )

    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v3.0 - 全球资产版）
        
        支持：A股(筹码/打板)、美股(业绩/回购)、加密货币(周期/链上)、商品(宏观/避险)
        """
        code = context.get('code', 'Unknown')
        asset_type = self._determine_asset_type(code)

        # 确保数据字典存在
        today = context.get('today') or {}
        trend = context.get('trend_analysis') or {}
        realtime = context.get('realtime') or {}
        
        # # 优先使用上下文中的股票名称
        # stock_name = context.get('stock_name', name)
        # if not stock_name or stock_name == f'股票{code}':
        #     stock_name = STOCK_NAME_MAP.get(code, f'{asset_type}:{code}')
        
        # 获取资产名称（优先级：context > 映射表）
        stock_name = context.get('stock_name') or name or f"Asset-{code}"
            
        # today = context.get('today', {})
        # trend = context.get('trend_analysis', {})
        

        # 安全获取当前价格（用于后续计算）
        current_price_raw = realtime.get('price') or today.get('close')
        try:
            current_price = float(current_price_raw) if current_price_raw is not None else None
        except (ValueError, TypeError):
            current_price = None
        
        # ========== 1. 头部与资产专属逻辑 ==========
        prompt = f"# {asset_type} 决策仪表盘分析请求\n\n"
        prompt += f"## 📊 基础信息\n| 代码 | 名称 | 类型 | 日期 |\n|---|---|---|---|\n| **{code}** | **{stock_name}** | {asset_type} | {context.get('date', '未知')} |\n\n"

        # 注入不同资产的分析“大脑”
        if asset_type == "ASHARE":
            prompt += """## 🧠 A股分析逻辑（核心）
1. **资金驱动**：高度关注主力资金流向、筹码集中度和获利盘比例。
2. **情绪博弈**：关注涨跌停板效应、连板高度和打板情绪。
3. **技术形态**：严格遵守 MA5/10/20 多头排列规则，乖离率过大严禁追高。
"""
        elif asset_type == "CRYPTO":
            prompt += """## 🧠 加密货币分析逻辑（核心）
1. **高波动性**：忽略微小的涨跌幅，重点关注 24小时成交量 和 关键支撑压力位互换。
2. **趋势指标**：短期看 MA5/10，牛熊分界看 MA50/MA200，忽略“涨跌停”概念。
3. **消息驱动**：重点关注 SEC 监管、ETF 流向、交易所动态、减半周期。
"""
        elif asset_type == "COMMODITY":
            prompt += """## 🧠 黄金/商品分析逻辑（核心）
1. **宏观定价**：必须结合 **美元指数(DXY)** 和 **美债收益率** 的反向关系进行分析。
2. **避险属性**：关注地缘政治冲突、全球央行购金动作。
3. **技术心理**：关注整数关口（如 2000, 2500）的心理支撑/压力。
"""
        else: # US_STOCK
            prompt += """## 🧠 美股分析逻辑（核心）
1. **基本面主导**：核心关注 EPS (每股收益)、财报指引 (Guidance) 和 股票回购 (Buyback)。
2. **机构行为**：关注成交量是否在关键技术位（如突破位）放大。
3. **大盘共振**：个股走势需结合纳指 (IXIC) / 标普500 (SPX) 的整体环境。
"""

        # ========== 2. 宏观上下文 ==========
        # if 'macro_indicators' in context:
            # m = context['macro_indicators']
        macro = context.get('macro_indicators')
        if isinstance(macro, dict):
            prompt += f"""
### 🌍 全球宏观定价环境
| 指标 | 当前数值 | 影响逻辑 |
|------|----------|----------|
| 美元指数(DXY) | {m.get('DXY', 'N/A')} | 走强则压制黄金/BTC |
| 10年期美债收益率 | {m.get('US10Y', 'N/A')}% | 走高则杀科技股估值 |
"""
        
        # ========== 3. 技术面数据 ==========
        prompt += f"""
---
## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} |
| 开盘价 | {today.get('open', 'N/A')} |
| 最高价 | {today.get('high', 'N/A')} |
| 最低价 | {today.get('low', 'N/A')} |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |


### 🕒 实时观测 （北京时间：{context.get('analysis_time', '未知')})
| 指标 | 实时数值 | 说明 |
|------|----------|------|
| **最新价格** | **{realtime.get('price', 'N/A')}** | 包含盘前/盘后价格 |
| 市场备注 | {realtime.get('market_note', '正常交易时段')} | |

### 均线系统
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期 |
| MA10 | {today.get('ma10', 'N/A')} | 短期 |
| MA20 | {today.get('ma20', 'N/A')} | 中期月线 |
| MA50 | {trend.get('ma50', 'N/A')} | 中期季度线 |
| MA200 | {trend.get('ma200', 'N/A')} | 长期牛熊分界线 |
| 均线形态 | {context.get('ma_status', '未知')} | |
"""
        # 长期主义过滤器逻辑加固
        ma200_raw = trend.get('ma200')
        if ma200_raw and current_price:
            try:
                ma200 = float(ma200_raw)
                dist_to_ma200 = (current_price - ma200) / ma200
                status = "🔴 长期走势破位" if dist_to_ma200 < 0 else "🟢 长期多头趋势"
                prompt += f"\n**⚖️ 长期过滤器**: 现价偏离 MA200 为 {dist_to_ma200:.2%} ({status})。\n"
            except (ValueError, TypeError): pass
            except: pass
                    
#         # 添加实时行情（如果有）
#         if realtime:
#             # rt = context['realtime']
#             prompt += f"""
# ### 实时指标
# | 指标 | 数值 |
# |------|------|
# | 现价 | {rt.get('price', 'N/A')} |
# | 量比 | {rt.get('volume_ratio', 'N/A')} |
# | 换手率 | {rt.get('turnover_rate', 'N/A')}% |
# | 市盈率 | {rt.get('pe_ratio', 'N/A')} |
# | 市值 | {self._format_amount(rt.get('total_mv'))} |
# """

        # ========== 4. 筹码分布 (仅 A股) ==========
        # ⚠️ 关键修改：只有 A股 才展示筹码数据，避免误导
        if asset_type == "ASHARE" and 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### 筹码分布（A股专用）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{profit_ratio:.1%}** | >90%需警惕回调 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} | 现价上方套牢盘需消化 |
| 90%集中度 | {chip.get('concentration_90', 0):.2%} | <10%为高度集中 |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        else:
            prompt += "\n**注：该资产不适用筹码分布分析，请勿臆造数据，JSON中相关字段请返回 null。**\n"

        # ========== 5. 趋势分析预判 ==========
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 高危" if trend.get('bias_ma5', 0) > 5 else "✅ 安全"
            prompt += f"""
### 趋势系统预判
| 指标 | 数值/状态 |
|------|------|
| 趋势强度 | {trend.get('trend_strength', 0)}/100 |
| **MA5乖离率** | **{trend.get('bias_ma5', 0):+.2f}%** ({bias_warning}) |
| 系统信号 | {trend.get('buy_signal', '未知')} |
| 风险因素 | {', '.join(trend.get('risk_factors', ['无']))} |
"""

        # ========== 6. 新闻情报 ==========
        prompt += "\n---\n## 📰 舆情情报\n"
        if news_context:
            prompt += f"""
搜索结果摘要（重点提取风险/利好/宏观）：    
{news_context}
"""
        else:
            prompt += "未搜索到近期相关新闻。\n"

        # ========== 7. 分析任务 (动态生成) ==========
        task_questions = ""
        if asset_type == "ASHARE":
            task_questions = """1. ❓ 是否满足 MA5>MA10>MA20 多头排列？
2. ❓ 筹码结构是否健康（获利盘是否过高）？
3. ❓ 乖离率是否过大（>5%）？"""
        elif asset_type == "CRYPTO":
            task_questions = """1. ❓ 当前价格相对于 MA50/MA200 的位置？
2. ❓ 成交量是否有显著放大（趋势确认）？
3. ❓ 消息面是否有监管或黑客利空？"""
        elif asset_type == "COMMODITY":
            task_questions = """1. ❓ 结合美元指数(DXY)走势，黄金是否具备上涨动力？
2. ❓ 全球避险情绪如何？
3. ❓ 技术面是否面临整数关口压力？"""
        else: # US_STOCK
            task_questions = """1. ❓ 公司基本面/财报预期如何？
2. ❓ 机构资金流向迹象？
3. ❓ 纳指/标普大盘环境是否配合？"""

        prompt += f"""
---
## ✅ 最终任务：生成决策仪表盘

请为 **{stock_name}** 生成 JSON 格式决策报告。

### 重点回答：
{task_questions}

### JSON 输出要求：
- **core_conclusion**: 核心结论（买入/卖出/观望）
- **trend_prediction**: 趋势预测（看多/看空/震荡）
- **operation_advice**: 操作建议（具体点位）
- **confidence_level**: 置信度（高/中/低）

请输出标准 JSON。"""
        
        return prompt
    
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"
    
    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 Gemini 响应（决策仪表盘版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)
                
                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)
                
                # 解析所有字段，使用默认值防止缺失
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震荡'),
                    operation_advice=data.get('operation_advice', '持有'),
                    confidence_level=data.get('confidence_level', '中'),
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技术面数据'),
                    success=True,
                )
            else:
                # 没有找到 JSON，尝试从纯文本中提取信息
                logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = '震荡'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '买入'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '卖出'
        
        # 截取前500字符作为摘要
        summary = response_text[:500] if response_text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，仅供参考',
            risk_warning='分析结果可能不准确，建议结合其他信息判断',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 Gemini 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
