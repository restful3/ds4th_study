"""
그래프 시각화 유틸리티

이 모듈은 LangGraph의 그래프를 시각화하는 함수를 제공합니다.
"""

from typing import Optional


def display_graph(graph, output_path: Optional[str] = None):
    """
    그래프를 시각화합니다.

    Args:
        graph: 컴파일된 LangGraph 그래프
        output_path: 이미지를 저장할 경로 (None이면 화면에 표시)

    Example:
        >>> graph = create_graph()
        >>> display_graph(graph)
        >>> display_graph(graph, "graph.png")
    """
    try:
        # Mermaid PNG 생성
        png_data = graph.get_graph().draw_mermaid_png()

        if output_path:
            # 파일로 저장
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"✅ 그래프 이미지 저장: {output_path}")
        else:
            # Jupyter/IPython 환경에서 표시
            try:
                from IPython.display import Image, display
                display(Image(png_data))
            except ImportError:
                print("⚠️  IPython을 사용할 수 없습니다.")
                print("   그래프를 보려면 output_path를 지정하여 파일로 저장하세요.")

    except Exception as e:
        print(f"❌ 그래프 시각화 실패: {e}")
        print("   graphviz가 설치되어 있는지 확인하세요:")
        print("   - macOS: brew install graphviz")
        print("   - Ubuntu: sudo apt-get install graphviz")
        print("   - Windows: https://graphviz.org/download/")


def print_graph_structure(graph):
    """
    그래프의 구조를 텍스트로 출력합니다.

    Args:
        graph: 컴파일된 LangGraph 그래프

    Example:
        >>> graph = create_graph()
        >>> print_graph_structure(graph)
    """
    try:
        graph_obj = graph.get_graph()

        print("=" * 60)
        print("그래프 구조")
        print("=" * 60)

        # 노드 출력
        print("\n📦 노드:")
        for node in graph_obj.nodes:
            print(f"  - {node}")

        # 엣지 출력
        print("\n🔗 엣지:")
        for edge in graph_obj.edges:
            print(f"  - {edge}")

        print()

    except Exception as e:
        print(f"❌ 그래프 구조 출력 실패: {e}")


def export_mermaid_code(graph) -> str:
    """
    그래프의 Mermaid 코드를 반환합니다.

    Args:
        graph: 컴파일된 LangGraph 그래프

    Returns:
        Mermaid 다이어그램 코드

    Example:
        >>> graph = create_graph()
        >>> mermaid = export_mermaid_code(graph)
        >>> print(mermaid)
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"그래프 변환 실패: {e}"


if __name__ == "__main__":
    """테스트 코드"""
    print("=" * 60)
    print("Visualization 유틸리티 테스트")
    print("=" * 60)

    # 간단한 테스트 그래프 생성
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        message: str

    def node1(state: State) -> State:
        return {"message": state["message"] + " -> node1"}

    def node2(state: State) -> State:
        return {"message": state["message"] + " -> node2"}

    graph = StateGraph(State)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    compiled_graph = graph.compile()

    # 구조 출력
    print_graph_structure(compiled_graph)

    # Mermaid 코드 출력
    print("📊 Mermaid 코드:")
    print(export_mermaid_code(compiled_graph))
    print()

    print("✅ 테스트 완료")
