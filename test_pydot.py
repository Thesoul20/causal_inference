import pydot
import os

causal_graph = """
digraph G {
    # 显式声明所有节点
    Age;
    DrugDosage;
    ExerciseHours;
    SodiumIntake;
    BloodPressure;

    # 然后定义边
    Age -> DrugDosage;
    Age -> ExerciseHours;
    Age -> BloodPressure;
    DrugDosage -> BloodPressure;
    ExerciseHours -> BloodPressure;
    SodiumIntake -> BloodPressure;
}
"""

print("Attempting to parse DOT string with pydot...")
try:
    # pydot.graph_from_dot_data 返回一个 pydot.Dot 对象的列表
    graphs = pydot.graph_from_dot_data(causal_graph)

    if graphs:
        graph = graphs[0] # 获取第一个图对象
        print("pydot successfully parsed the graph.")
        print(f"Graph nodes: {[node.get_name() for node in graph.get_nodes()]}")
        print(f"Graph edges: {[edge.get_source() + ' -> ' + edge.get_destination() for edge in graph.get_edges()]}")

        # 尝试生成一个图片文件，这需要Graphviz的dot可执行文件
        output_file = "test_causal_graph.png"
        graph.write_png(output_file)
        print(f"Graph successfully written to {output_file}")
        if os.path.exists(output_file):
            print(f"File {output_file} exists, which means Graphviz was used successfully.")
        else:
            print(f"Warning: File {output_file} was not created, even after pydot reported success. Check Graphviz installation path.")

    else:
        print("pydot parsed the string but returned an empty list of graphs. This is unexpected for valid DOT.")
        print("This might indicate an issue with the DOT string itself or pydot's interpretation.")

except Exception as e:
    print(f"pydot failed to parse the graph or write the image. Error: {e}")
    print("This indicates a problem with pydot or its interaction with Graphviz.")


