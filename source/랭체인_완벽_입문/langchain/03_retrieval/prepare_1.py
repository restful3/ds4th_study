from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./sample.pdf") #← sample.pdf 로드
documents = loader.load()

print(f"문서 개수: {len(documents)}") #← 문서 개수 확인
print(f"첫 번째 문서의 내용: {documents[0].page_content}") #← 첫 번째 문서의 내용을 확인
print(f"첫 번째 문서의 메타데이터: {documents[0].metadata}") #← 첫 번째 문서의 메타데이터를 확인
