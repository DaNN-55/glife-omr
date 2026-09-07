# 吉他六线谱 OMR：复杂截图、合成训练与人工核验的替代路径

日期：2026-09-06。本文是方案调研，不启动训练、不改变词表或模型。

## 已确认的项目前提

Glife 调用的 `kk9293/guitar-tab-omr` 是把**裁切后的单个六线谱 system**转为 Tabbox tokens 的模型；模型卡明确排除了任意扫描谱和整页输入。它在 renderer-augmented 验证集上的 200 个样本中，exact match 为 76.5%，parse success 为 99.5%：能解析不代表弦、品、节奏和技巧均正确。[模型卡](https://huggingface.co/kk9293/guitar-tab-omr) [训练仓库](https://github.com/LIU9293/guitar-tab-omr)

当前三张“透明谱”截图的 RGBA alpha 均为 255：透明效果已经和人物背景烘焙为普通 RGB 像素，不能通过读取 alpha 通道恢复白底。当前输入缩放是保持比例后补白至 1200×224，不是强行拉伸。因此首要问题是目标域（视频叠层、背景、压缩、技巧密度）与清爽数字渲染训练分布不一致，不是缩放变形。

结论：不必先走“全部人工逐张标注再重训”。先把输入规范化、用六线几何和乐谱结构挡住明显错误，并将人工只投到高价值样本；若仍不足，再用结构化乐谱自动生成真值做微调。不能保证仅靠前处理把已烘焙的透明叠层还原成无损白底。

## 不训练或只改流程的优先顺序

| 路径 | 对当前错误的作用 | 成功条件与边界 |
| --- | --- | --- |
| 1. 找结构化来源 | 有原始 GP/GPX/MusicXML、可保留语义的网页谱数据时，直接导入或作为真值，跳过截图 OMR；PDF/SVG 若仅保留图形，则只能改善输入或辅助提取，不能默认已有完整乐谱语义 | 仅限你有权获取、并确实存在的结构化来源；不能从一张像素截图反推原文件。 |
| 2. 单 system 定位、矫正和六线几何 | 先检测长水平线，得到 6 条弦的中心线、间距、倾斜角与 system 边界；据此 deskew、裁切，且把音符候选按 y 投影到最近弦 | Hough 变换能检测图中直线，OpenCV 也有线段输出；但背景中的水平物体会造成假线，必须以“6 条近等距、同角度、共同 x 区间”筛选并保留失败回退。[OpenCV Hough 文档](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html) |
| 3. 多帧共识输入（只做 POC） | 对同一谱面、谱面位置可对齐且人物背景在变化的连续帧：先按六线几何对齐，再以逐像素分位数/局部方差生成候选掩膜或候选增强图，和原帧分别送 OMR，比较识别与结构差异，分歧交人工；结构评分较高不等于识别正确 | OpenCV 的背景减除明确面向静态相机和背景模型；本场景是静态谱叠在变化背景上，不能直接套用并声称“去背景”。未知 alpha 和已烘焙像素使无损恢复不可辨识；必须在留出样本上证明收益。[OpenCV 背景减除文档](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html) |
| 4. 多视图推理与拒答 | 对原图、对比度/去阴影图、几何裁切图分别推理。只有相同或高一致的 token 片段自动接受；分歧片段、低置信片段和无法闭合的小节进人工核验 | 这减少“看起来可导出但实际错”的自动放行，不会凭空找回已看不清的数字或技巧。 |
| 5. 结构校验作为拦截器 | 用已知六弦 y 几何、弦号范围、品格范围、bar/beat/duration 闭合、技巧与相邻音关系检查；违例时标为 `needs_review`，不要用规则猜测替换品或弦 | 约束可发现无效谱，不能证明剩余谱正确。长 slide、bend、hammer-on 等应保持为音符间关系/token，不能降格为孤立字符框；复杂谱面对象本来就强依赖上下文。[DeepScoresV2 论文](https://stdm.github.io/downloads/papers/ICPR_2020.pdf) |
| 6. 外部引擎只作对照基线 | Soundslice 官方页面确认能扫描 TAB/PDF/图片，适合挑一小批有授权样本做“是否节省人工”的盲测 | 它对 tab-only 将所有音符先导为四分音符，只假设标准六弦调弦，且 bends、hammer-on/pull-off、harmonics、palm mute、slides、vibrato 均不识别；不能当作技巧谱到 GP 的替代方案。[官方限制](https://www.soundslice.com/help/en/creating/pdf-import/294/supported-notations/) [tab-only 限制](https://www.soundslice.com/help/en/creating/pdf-import/297/tab-only-music/) |

最小实现建议是 2→4→5：保持现有 Tabbox 模型和人工工作台，不引入新模型或第三方依赖；先建立一个从未参与规则调整的困难集，再决定是否值得训练。第 3 项只有拿到同一谱面连续帧时再做。

## 如果要训练：推荐的混合数据流程

```text
可编辑、获授权的 GP/GPX/MusicXML/AlphaTex
       ↓ 解析为同一版本的 Tabbox tokens
多种 TAB 渲染器/布局 → 基础图 + 自动真值
       ↓ 域随机化：缩放、压缩、模糊、摩尔纹、亮度、透视、
                    半透明叠层与已授权背景合成
合成困难训练集 ───────┐
人工复核过的真实截图 ──┼→ 按来源分组切分 train / val / hidden test
保留独立验证与测试集，均不参与梯度训练
       ↓
从现有 checkpoint 微调 → 原图/增强图 A/B → 逐项错误分析 → 保留或回滚
```

### 哪些素材可不靠逐图人工标注

1. **符号真值和基础图**：从你已获授权的结构化吉他谱导出 tokens，再以 Guitar Pro、alphaTab 或 MuseScore 类渲染器输出单 system 图。图和 token 来自同一符号源，可自动配对；仍须检查渲染器是否实际显示了标签里的每种技巧，隐藏属性不能直接充当可见图像标签。
2. **视觉域随机化**：在基础图上自动生成不同字体、间距、系统宽度、裁切、JPEG/视频压缩、模糊、噪声、亮度梯度、轻微透视和 screen moiré。对“透明谱”要用自己的/获授权背景做 alpha 合成，再压平为 RGB；不要把真实截图背景当作可随意再分发的训练资产。
3. **技巧组合覆盖**：由结构化谱程序组合音符和技巧 token，渲染为不同密度、跨小节和相邻音关系的例子。产生后要运行 token→预览→结构校验，避免渲染器/转换器的 bug 成为标签。
4. **弱标签候选**：现有模型高置信且结构通过的真实样本可排队，但在人工确认前不是训练真值；不能把模型自己的输出直接回灌训练。

这条路线有可复用的 OMR 先例：Camera-PrIMuS 从 PAEC 的符号源经 Verovio 自动渲染 PNG/SVG，再加字体、旋转、噪声、阴影、形变和运动模糊，得到相机域训练样本及自动真值。[Camera-PrIMuS 原论文](https://archives.ismir.net/ismir2018/paper/000033.pdf) MuseScore 的 OMR benchmark 也以可编辑符号谱为真值，生成带纸纹、划痕和旋转增强的 PDF 图像。[官方仓库](https://github.com/musescore/omr_benchmark) 这两者是普通五线谱，不是六线谱；可借用生成方法，不能直接训练或证明 TAB 的弦/品/技巧性能。

SynthTab 则展示了同一原则在吉他任务上的可扩展性：从带指法/技巧信息的符号 TAB 合成数据，原符号 TAB 就是标签；作者也公开记录了渲染 bug，说明自动真值仍必须用独立验证和抽检约束。[SynthTab 论文](https://arxiv.org/pdf/2309.09085.pdf) [官方仓库及已知 bug](https://github.com/yongyizang/SynthTab)

### 人工应该标什么、按什么顺序标

人工不是消失，而是变成三个小而清楚的集合：

1. **隐藏测试集**：先冻结，不进训练。按来源/视频、背景类型、清爽/烘焙叠层、技巧类型和密度分层，逐个确认完整 tokens。
2. **真实微调集**：来自同一目标域的经复核截图；每条需保存原图、正确 tokens、标签规则版本、来源族和错误类别。相邻视频帧和同一首谱的近重复图必须归同一组，不能分别落到 train 与 test。
3. **主动学习队列**：每轮只人工看“模型最不确定 + 和已有样本最不相似 + 技巧稀少”的一批。端到端 token 模型可先用 greedy/beam 分歧、token entropy、约束修复量、弦线几何偏离和预测/人工差异作为实用排序分数。主动学习的基本闭环是：先少量训练、选不确定未标样本、人工标、加入训练、复训；它能减少标注量，但收益必须在 Glife 的隐藏困难集上验证。[不确定性采样研究](https://epub.ub.uni-muenchen.de/91888/) [图像主动学习原论文](https://arxiv.org/abs/1703.02910)

## 样本量：仅作首轮规划估计，不是已验证门槛

现有基础模型已经在约 5 万张清爽 renderer-augmented 样本上训练；此处的目标是补齐目标域，不能只堆同一种清爽合成图。建议以“独立 system 数”计数而非视频帧数：

| 集合 | 首轮规划量 | 构成 |
| --- | ---: | --- |
| 冻结困难测试 | 150–250 个真实、逐条复核 system | 至少覆盖清爽/背景/烘焙叠层和常见/稀少技巧；不训练、不调阈值。 |
| 合成训练 | 2,000–5,000 个 system 图 | 约 200–500 个不同结构化谱片段，每片生成多种布局和背景条件；以目标域难例为主。 |
| 真实微调 | 300–800 个逐条复核 system | 不同来源族；复杂背景和技巧密集样本需过采样。 |
| 每个未达标错误簇 | 先补 50–100 个**独立**真值 system，再复测 | 例如 5/6 弦混淆、两位数品格、slide/bend、hammer/pull、vibrato 各自建簇；没有足够错误簇证据，不扩大词表。 |

这些数字是为了决定第一轮是否有信号的预算，不能推导准确率。若测试集的主要错误是“符号根本不可见”，增加训练样本无效，应回到采集/多帧/结构化来源；若是视觉可见但系统性误读，才做合成+真实微调。每轮用独立验证集选择模型；最终再用冻结真实测试集确认是否改善弦-品事件、节奏/小节闭合、技巧关系以及人工修正时间的 checkpoint；任一关键项退化即回滚。

## 训练实施的最小闭环

1. 冻结标签规则、词表和 150–250 个真实测试 system；另留约 100 个真实 system 作调参与模型选择的验证集。测试集不用于反复选择模型和改规则。先记录当前 checkpoint 的分项基线。
2. 从获授权的结构化谱自动导出图/tokens，生成目标域视觉增强；人工只复核真实集和合成管线的抽样。
3. 加载现有 OMR checkpoint，以合成与真实样本混合微调，再用真实微调集短轮微调；保持按“来源族”分割，记录随机种子、数据版本和 checkpoint。
4. 同时跑原图、增强图和现有 checkpoint 的 A/B；输出按 string/fret、duration/measure、technique relation、parse success 和人工修正分钟数拆分的报告。
5. 把失败样本按主动学习队列送审核，重复 2–4；若新增技巧，须先完成标签定义、词表与输出层兼容改造，再生成对应标签和训练；旧词表无法学习不存在的 token。

## 来源与证据边界

1. [Tabbox 模型卡](https://huggingface.co/kk9293/guitar-tab-omr)：当前模型的 intended use、训练域、架构和限制；为作者自报指标，未给透明叠层分项。
2. [Tabbox 官方训练仓库](https://github.com/LIU9293/guitar-tab-omr)：当前训练命令、输入尺寸和 200 个 held-out 样本的公开指标；不是复杂截图基准。
3. [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)：直线/线段检测 API 的能力证据；不保证六线谱定位成功。
4. [OpenCV Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)：背景模型适用条件；恰好说明它不能直接保证解决静态谱叠在动态背景上的已合成截图。
5. [Camera-PrIMuS](https://archives.ismir.net/ismir2018/paper/000033.pdf)：从结构化谱自动渲染并做相机退化的 OMR 生成方法；任务不是 TAB。
6. [MuseScore OMR Benchmark](https://github.com/musescore/omr_benchmark)：符号谱→增强图像→可评测真值的公开工作流；任务不是 TAB。
7. [SynthTab](https://arxiv.org/pdf/2309.09085.pdf)：吉他结构化 TAB 可合成并提供真值的证据；论文做的是音频 TAB 转写，视觉渲染需要本项目另行实现和验证。
8. [Soundslice PDF/image TAB 支持与限制](https://www.soundslice.com/help/en/creating/pdf-import/294/supported-notations/)：外部对照工具的官方能力边界，不等于它可提供本地 GP 结果或完整技巧真值。
9. [Nguyen et al., uncertainty sampling](https://epub.ub.uni-muenchen.de/91888/) 和 [Gal et al., Deep Bayesian Active Learning](https://arxiv.org/abs/1703.02910)：把人工标注集中在不确定样本的原理；没有证明其在 Glife token OMR 上必然节省多少人工。

## 本地代码核验补充

- `data/omr-records/manifest.json` 当前仅 1 条有效训练候选；保存过审核不代表进入候选集。
- `data/omr-token-gaps.json` 与当前词表确认缺少 P.M. 与前置装饰音。训练前要统一标签、解码、时值校验和 AlphaTex 转换，不能仅增加图片。
- `vendor/guitar-tab-omr/scripts/guitar_omr_train.py` 当前创建模型后直接优化，没有加载既有 OMR checkpoint 的入口；`--pretrained-encoder` 只控制视觉骨干，不能当作 OMR 微调。后续须增加权重加载和兼容性检查，保持旧 token ID；新增 token 时迁移 embedding/输出层等对应参数。
- 同脚本当前随机按样本拆分。须改为按原曲/视频/谱源分组，先分组再增强；同源帧、同源不同渲染均不可跨训练/验证/测试。
- `guitar_omr_dataset_smoke.py:encode_tokens` 会截断过长标签。训练前须统计长度并拒绝静默截断；按完整小节裁切时保留拍号、调弦及跨小节技巧上下文。
- `review/omr-workbench.html` 直接将 tokens 转 AlphaTex 后导出 GP，已有 `7 - string` 弦序映射与未转换技巧警告。须分别评估图片到 tokens 与 tokens 到 GP，避免把转换错误当模型错误。
- `data/omr-ab-001/step5b-guitar-tab-omr-ab.json` 是旧单样本 A/B：恢复图的小节时值闭合，但仍有品位替换和多音。该样本不是独立测试，也未评分技巧，不能证明增强有效或全局准确率。
