# Guitar Tab OMR Review

This context names the score material and human review records used to improve guitar tablature recognition.

## Language

**Score Sample（谱面样本）**:
一次独立识别和人工核验所针对的一张单系统谱面图像。整页或整首曲谱由多个 Score Sample 组成，不作为单个样本。
_Avoid_: 整页样本、整曲样本

**Training Candidate（训练候选）**:
由 Score Sample、模型原始预测和人工修正标签组成，已通过当前词表与数据格式检查、但尚未用于训练的记录。
_Avoid_: 训练数据、已训练样本

**Candidate Dataset（候选数据集）**:
由当前有效的 Training Candidate 组成、可供训练流程读取的数据集合。它通过数据检查，但不表示其中记录已经参与训练。
_Avoid_: 训练集、已批准训练集

**Superseded Candidate（已取代候选）**:
因人工标签需要纠正而被新版 Training Candidate 取代的不可变记录。它保留用于追溯，但不得进入后续训练。
_Avoid_: 已删除样本、旧训练数据

**Prediction Record（识别记录）**:
某个模型 checkpoint 对 Score Sample 产生的原始预测及其识别配置。它可用于模型效果对比，但不是 Training Candidate，也不会因同一正确标签而重复加入 Candidate Dataset。
_Avoid_: 训练样本、正确标签

**Label Schema（标签模式）**:
定义人工修正标签所使用的 token 词表与语法的版本化规则。同一个 Candidate Dataset 只能包含兼容 Label Schema 的 Training Candidate。
_Avoid_: 模型版本、checkpoint 版本
