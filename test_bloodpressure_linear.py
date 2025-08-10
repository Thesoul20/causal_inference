import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx

class SimpleDowhyCase():
    def __init__(self, random_seed=42, show_falg: bool=False):
        np.random.seed(random_seed)
        self.show_falg = show_falg

    def generate_test_data(self) -> pd.DataFrame:
        # This method would generate test data for the case
        
        # --- 1. 生成模拟数据 ---
        print("--- 1. 生成模拟数据 ---")
        num_samples = 1000

        age = np.random.normal(55, 10, num_samples)
        age = np.clip(age, 30, 80)

        exercise_hours = 10 - 0.15 * age + np.random.normal(0, 1.5, num_samples)
        exercise_hours = np.clip(exercise_hours, 0, 15)

        drug_dosage = 5 + 0.1 * age + np.random.normal(0, 2, num_samples)
        drug_dosage = np.clip(drug_dosage, 0, 20)

        sodium_intake = np.random.normal(2500, 500, num_samples)
        sodium_intake = np.clip(sodium_intake, 1000, 4000)

        blood_pressure = (
            80
            + 0.5 * age
            - 2.0 * drug_dosage
            - 1.5 * exercise_hours
            + 0.01 * sodium_intake
            + np.random.normal(0, 5, num_samples)
        )
        blood_pressure = np.clip(blood_pressure, 90, 180)

        data = pd.DataFrame({
            'Age': age,
            'ExerciseHours': exercise_hours,
            'DrugDosage': drug_dosage,
            'SodiumIntake': sodium_intake,
            'BloodPressure': blood_pressure
        })

        print("模拟数据前5行：")
        print(data.head())
        print("\n数据描述：")
        print(data.describe())
        return data

    def define_Causal_Graph(self,) -> nx.DiGraph:
        # This method would define the causal graph for the case
        # 现在我们使用 networkx 构建图
        print("\n--- 2. 定义因果图 (使用 networkx.DiGraph) ---")
        causal_graph_nx = nx.DiGraph() # 创建一个有向图对象

        # 添加节点
        causal_graph_nx.add_nodes_from(['Age', 'DrugDosage', 'ExerciseHours', 'SodiumIntake', 'BloodPressure'])

        # 添加有向边
        causal_graph_nx.add_edge('Age', 'DrugDosage')
        causal_graph_nx.add_edge('Age', 'ExerciseHours')
        causal_graph_nx.add_edge('Age', 'BloodPressure')
        causal_graph_nx.add_edge('DrugDosage', 'BloodPressure')
        causal_graph_nx.add_edge('ExerciseHours', 'BloodPressure')
        causal_graph_nx.add_edge('SodiumIntake', 'BloodPressure')

        print("NetworkX 图的节点:", list(causal_graph_nx.nodes))
        print("NetworkX 图的边:", list(causal_graph_nx.edges))
        return causal_graph_nx

    #### 案例 A：分析 `DrugDosage` 对 `BloodPressure` 的因果效应
    def _caseA(self, data: pd.DataFrame, causal_graph_nx: nx.DiGraph, show_falg: bool=False):
        print("\n\n--- 案例 A: 分析 DrugDosage 对 BloodPressure 的因果效应 ---")

        # 3.1. 模型构建 (Model)
        model_drug = CausalModel(
            data=data,
            graph=causal_graph_nx,
            treatment='DrugDosage',
            outcome='BloodPressure'
        )

        # 打印模型概览，并尝试生成因果图的PDF（如果Graphviz已安装）
        print("\nDrugDosage 模型概览:")
        if show_falg:
            model_drug.view_model()


        # 3.2. 识别 (Identify)
        identified_estimand_drug = model_drug.identify_effect(proceed_when_unidentifiable=True)
        print("\n识别到的估计量 (Identified Estimand) for DrugDosage:")
        print(identified_estimand_drug)

        # 3.3. 估计 (Estimate)
        estimate_drug = model_drug.estimate_effect(
            identified_estimand_drug,
            method_name="backdoor.linear_regression"
        )
        print("\n估计结果 (Estimate) for DrugDosage:")
        print(estimate_drug)
        print(f"结论：药物剂量每增加1单位，血压平均改变: {estimate_drug.value:.2f} 单位")

        # 3.4. 反驳 (Refute) - 检验结果的稳健性
        print("\n反驳测试 (Refutation) for DrugDosage:")

        # **API 修正：使用 refute_estimate**
        refute_result_placebo_drug = model_drug.refute_estimate( # <--- 关键修改在这里
            identified_estimand_drug,
            estimate_drug,
            method_name="placebo_treatment_refuter"
        )
        print("\n安慰剂处理反驳结果 (Placebo Treatment Refuter):")
        print(refute_result_placebo_drug)

        # **API 修正：使用 refute_estimate**
        refute_result_random_common_cause_drug = model_drug.refute_estimate( # <--- 关键修改在这里
            identified_estimand_drug,
            estimate_drug,
            method_name="random_common_cause"
        )
        print("\n随机共同原因反驳结果 (Random Common Cause Refuter):")
        print(refute_result_random_common_cause_drug)

    #### 案例 B：分析 `ExerciseHours` 对 `BloodPressure` 的因果效应
    def _caseB(self, data: pd.DataFrame, causal_graph_nx: nx.DiGraph, show_falg: bool=False):
        print("\n\n--- 案例 B: 分析 ExerciseHours 对 BloodPressure 的因果效应 ---")

        # 3.1. 模型构建 (Model)
        model_exercise = CausalModel(
            data=data,
            graph=causal_graph_nx,
            treatment='ExerciseHours',
            outcome='BloodPressure'
        )
        if show_falg:
            model_exercise.view_model()


        # 3.2. 识别 (Identify)
        identified_estimand_exercise = model_exercise.identify_effect(proceed_when_unidentifiable=True)
        print("\n识别到的估计量 (Identified Estimand) for ExerciseHours:")
        print(identified_estimand_exercise)

        # 3.3. 估计 (Estimate)
        estimate_exercise = model_exercise.estimate_effect(
            identified_estimand_exercise,
            method_name="backdoor.linear_regression"
        )
        print("\n估计结果 (Estimate) for ExerciseHours:")
        print(estimate_exercise)
        print(f"结论：运动时长每增加1小时，血压平均改变: {estimate_exercise.value:.2f} 单位")

        # 3.4. 反驳 (Refute) - 检验结果的稳健性
        print("\n反驳测试 (Refutation) for ExerciseHours:")

        # **API 修正：使用 refute_estimate**
        refute_result_placebo_exercise = model_exercise.refute_estimate( # <--- 关键修改在这里
            identified_estimand_exercise,
            estimate_exercise,
            method_name="placebo_treatment_refuter"
        )
        print("\n安慰剂处理反驳结果 (Placebo Treatment Refuter):")
        print(refute_result_placebo_exercise)

        # **API 修正：使用 refute_estimate**
        refute_result_random_common_cause_exercise = model_exercise.refute_estimate( # <--- 关键修改在这里
            identified_estimand_exercise,
            estimate_exercise,
            method_name="random_common_cause"
        )
        print("\n随机共同原因反驳结果 (Random Common Cause Refuter):")
        print(refute_result_random_common_cause_exercise)

    #### 案例 C：分析 `SodiumIntake` 对 `BloodPressure` 的因果效应
    def _caseC(self,data: pd.DataFrame, causal_graph_nx: nx.DiGraph, show_falg: bool=False):
        print("\n\n--- 案例 C: 分析 SodiumIntake 对 BloodPressure 的因果效应 ---")

        # 3.1. 模型构建 (Model)
        model_sodium = CausalModel(
            data=data,
            graph=causal_graph_nx,
            treatment='SodiumIntake',
            outcome='BloodPressure'
        )
        if show_falg:
            model_sodium.view_model()

        # 3.2. 识别 (Identify)
        identified_estimand_sodium = model_sodium.identify_effect(proceed_when_unidentifiable=True)
        print("\n识别到的估计量 (Identified Estimand) for SodiumIntake:")
        print(identified_estimand_sodium)

        # 3.3. 估计 (Estimate)
        estimate_sodium = model_sodium.estimate_effect(
            identified_estimand_sodium,
            method_name="backdoor.linear_regression"
        )
        print("\n估计结果 (Estimate) for SodiumIntake:")
        print(estimate_sodium)
        print(f"结论：钠摄入量每增加1单位，血压平均改变: {estimate_sodium.value:.4f} 单位")

        # 3.4. 反驳 (Refute) - 检验结果的稳健性
        print("\n反驳测试 (Refutation) for SodiumIntake:")

        # **API 修正：使用 refute_estimate**
        refute_result_placebo_sodium = model_sodium.refute_estimate( # <--- 关键修改在这里
            identified_estimand_sodium,
            estimate_sodium,
            method_name="placebo_treatment_refuter"
        )
        print("\n安慰剂处理反驳结果 (Placebo Treatment Refuter):")
        print(refute_result_placebo_sodium)

        # **API 修正：使用 refute_estimate**
        refute_result_random_common_cause_sodium = model_sodium.refute_estimate( # <--- 关键修改在这里
            identified_estimand_sodium,
            estimate_sodium,
            method_name="random_common_cause"
        )
        print("\n随机共同原因反驳结果 (Random Common Cause Refuter):")
        print(refute_result_random_common_cause_sodium)

    def run_test(self):
        # This method would run the test case
        data = self.generate_test_data()
        causal_graph_nx = self.define_Causal_Graph()
        self._caseA(data, causal_graph_nx, self.show_falg)
        self._caseB(data, causal_graph_nx, self.show_falg)
        self._caseC(data, causal_graph_nx, self.show_falg)
        pass
    
    @classmethod
    def run(cls, random_seed: int=42, show_falg: bool=False):
        # This method would run the test case
        instance = cls(show_falg=show_falg, random_seed=random_seed)
        instance.run_test()

if __name__ == '__main__':
    SimpleDowhyCase.run(show_falg=True)
