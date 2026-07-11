---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 부록 C 구조화된 소스에서 지식 그래프 구축하기


이 부록에서는 구조화된 데이터 소스에서 자신만의 지식 그래프 (knowledge graph, KG)를 구축하는 방법을 설명합니다. 이 책의 여러 곳에서 그러하듯이, 여기서도 생의학 활용 사례에 초점을 맞춥니다. 여기서는 microRNA-질병 연관성의 탐지입니다. 그림 C.1

![](images/7f9c857e6a536699b0b06c8098b89686f238d3574f05ec62a555a74d8dac344d.jpg)  
그림 C.1 질병(셀리악병)과 microRNA 간 관계의 예

는 우리가 구축할 지식 그래프의 핵심을 보여줍니다. 이 프로젝트에서는 2장에서 소개한 CRISP-DM 모델을 사용할 것입니다(그림 C.2 참조).

![](images/75eac3deae112f4a94df0240d6785636c54444b6af065e30b17096f993fbdc3d.jpg)  
그림 C.2 지식 그래프에 맞게 조정한 CRISP-DM 모델

### C.1 마이크로RNA–질병 연관: 워밍업


마이크로RNA (microRNA)–질병 연관은 생의학 분야에서 지식 그래프에 매우 적합한 활용 사례입니다. 이 절에서는 마이크로RNA가 무엇인지, 그리고 그것이 질병과 어떻게 연결되는지에 대한 간략한 생물학적 설명을 제공합니다. 그런 다음 이러한 목표를 달성하기 위한 비즈니스 목표와 우리가 이용할 수 있는 데이터를 개괄합니다.

### C.1.1 핵심 개념


마이크로RNA(이하 miRNA)는 비교적 최근에 발견된 비암호화 RNA (noncoding RNA)의 한 유형입니다(즉, 단백질로 번역되지 않는 RNA) [1]. 이러한 매우 작은 분자(19~22개의 뉴클레오타이드로 구성됨)는 상보적인 전령 RNA (messenger RNA, mRNA; 단백질로 번역되어야 하는 RNA)를 방해하여, 이른바 유전자 침묵 (gene silencing), 즉 유전자 발현의 조절 또는 유전자 발현에 대한 간섭을 유발합니다. miRNA는 번역 억제 (translational repression)와 mRNA 불안정화 (mRNA destabilization)의 조합을 통해 침묵을 수행합니다 [2]. 그림 C.3은 정상적인 인코딩이 어떻게 작동하는지를 보여 주며, 그림 C.4는 miRNA가 그것에 어떻게 영향을 미치는지를 보여 줍니다.

![](images/e368469df8565f5c24ffd1fbabe8a60cc72af8e4e3b765812bebcfb4ef1ba541.jpg)  
그림 C.3 DNA에 인코딩된 정보는 mRNA로 전사된 다음, 합성된 단백질을 구성하는 아미노산 사슬로 번역됩니다.

![](images/a19a62efc57c0a8ddc49ff0255e801389bf3165dfeaaa3b0e67a869aa4a5b287.jpg)  
그림 C.4 miRNA 서열은 너무 작아서 단백질로 번역될 수 없지만, 특정 mRNA를 표적으로 삼기에는 충분히 큽니다. 이 일이 발생하면 mRNA가 리보솜을 통과해 흐를 수 없으며, 단백질 합성이 일어날 수 없습니다.

연구들은 miRNA가 세포 분화 [5], 증식 [6], 신호 전달 [7], 바이러스 감염 [8] 등 많은 중요한 생물학적 과정 [3, 4]에 관여한다는 것을 보여 주었습니다. 새롭게 축적되는 증거들은 또한 miRNA가 암과 대사 질환 같은 복잡한 인간 질병의 발병기전 (pathogenesis)에 관련되어 있음을 시사합니다 [9–15]. 예를 들어, 연구자들은 mir-433 miRNA가 알려진 종양 관련 단백질인 GRB2의 발현을 조절함으로써 위암에 관여한다는 사실을 발견했습니다 [16]. 이 예제는 바로 이 측면에 초점을 맞출 것입니다.

### C.1.2 비즈니스 이해


miRNA와 질병 사이의 연결을 예측하고자 한다고 가정해 보겠습니다. 가능한 조합의 수는 방대합니다. 수천 종의 miRNA가 확인되었으며, 그 수와 영향은 이전에 생각했던 것보다 더 클 가능성이 높습니다 [17]. 또한 가정을 검증하기 위한 체외 실험은 비용이 많이 듭니다. miRNA와 질병 사이의 상관관계를 예측할 수 있다면 연구자들이 가장 가능성이 높은 대상에 연구를 집중하도록 도울 수 있으며 [18], 이러한 발견을 하는 데 필요한 비용과 시간을 줄일 수 있습니다.

이 시나리오는 지능형 조언자 시스템 (intelligent advisor system)의 개발과 관련이 있습니다(책 전반에서 논의됩니다). 비즈니스 목표는 또한 지식 그래프 (KG)에 대해 수행할 분석의 유형과 최종 사용자에게 제공할 “조언”의 유형을 결정합니다. 여기서 우리의 초점은 구축 단계에 있지만, 필요한 분석을 이해하면 그래프 모델을 개선하는 데 도움이 되므로 프로세스의 초기 단계에서 중요한 고려 사항입니다.

참고 우리가 시연하는 접근법과 기법은 어떤 시나리오에도 쉽게 적용될 수 있습니다. 우리의 목표는 기존의 구조화된 데이터 소스에서 시작하여 KG를 구축하는 방법을 보여 주는 것입니다.

### C.1.3 데이터 이해


miRNA에 관한 연구는 비교적 최근에 시작되었지만, 이 연구 분야에서 이용 가능한 정보의 양은 방대하며 쉽게 접근할 수 있습니다. Tools4miRs(https://tools4mirs.org) [19]라는 플랫폼은 “miRNA를 분석하는 데 필요한 모든 도구”를 제공한다고 주장합니다. 이 플랫폼은 170개가 넘는 방법과 많은 데이터베이스를 제공합니다.

우리는 우리의 목적과 관련된 데이터 소스를 선택했습니다. 사용된 각 데이터베이스에 대해 설명하고 링크를 제공하므로, 새 버전이 제공되는지 확인할 수 있습니다. 이 분야에 관심이 있고 KG를 확장하고자 한다면, Tools4miRs를 출발점으로 사용할 것을 강력히 권장합니다.

### C.2 miRNA 지식 그래프 구축


우리가 구축하려는 지식 그래프 (KG)는 miRNA와 병리 사이의 연결이 지닌 복잡성을 포착해야 합니다. 이를 통해 풍부한 데이터를 사용하고 누락된 링크, 즉 아직 발견되지 않은 연결을 예측하도록 학습할 수 있는 기계 학습 (ML) 모델을 설계할 수 있어야 합니다. 여기서는 예측 알고리즘을 다루지 않지만, 이러한 유형의 KG 구축이 가능하게 할 기회를 이해하는 것은 중요합니다.

우리는 miRNA와 질병에 관련된 사용 가능한 모든 데이터셋을 수집하는 것에서 시작한 다음, ML 모델이 분석을 처리하도록 할 수도 있습니다. 이러한 “탐욕적” 접근법은 데이터가 많을수록 주제의 복잡성을 더 잘 이해하게 된다고 가정하며, 이는 다시 ML 모델이 miRNA와 질병 사이의 관계를 지배하는 근본 규칙을 발견할 기회를 더 많이 제공해야 한다는 뜻입니다.

이론적으로는 이것이 맞지만, 실제로 모든 것을 KG에 쏟아 넣는 것은 현명한 선택이 아닙니다.

모든 데이터셋에는 신호 대 잡음비가 있습니다. 좋은 데이터셋에서도, 그 데이터셋이 링크 예측과 약간만 관련되어 있다면 잡음이 신호의 이점을 쉽게 압도할 수 있습니다.

새로운 데이터셋은 그래프 ML 알고리즘이 새로운 정보를 충분히 활용할 수 있기 전에 기존 KG와 조정되어야 합니다. 때로는 조정 과정이 단순하지 않으며, 그 결과로 생성되는 관계가 오류의 영향을 받을 수 있고, 이는 다시 새로운 데이터셋과 관련된 잡음을 증폭시킬 수 있습니다.

우리는 오히려 우리의 과제, 즉 병리 링크 예측에 도움이 될 miRNA 관련 데이터셋을 선택하는 것에서 시작하고자 합니다. 이러한 접근법은 상대적으로 복잡도가 낮은 KG를 만들어 내며, 이는 추론하기가 더 쉽습니다. 이 KG에서 나온 링크 예측 결과는 기준선으로 사용할 수 있으므로, 새로운 데이터 소스가 모델을 어떻게 개선할 수 있는지 정량화할 수 있습니다.

### C.2.1 알려진 miRNA–질병 연결 가져오기


먼저 miRNA와 질병 간의 알려진 연결을 포함하는 데이터셋을 식별하는 것부터 시작하겠습니다. 이 첫 번째 데이터 적재 (ingestion) 라운드를 위해 우리가 선택한 데이터 소스는 Human miRNA Disease Database (HMDD; http://www.cuilab.cn/hmdd) [20–22], Database of Differentially Expressed miRNAs in Human Cancers (dbDEMC; https://www.biosino.org/dbDEMC) [23], 그리고 miR2Disease (www.mir2disease.org) [24]입니다. 이러한 데이터셋은 서로 다른 출처에서 왔으며 다양한 연구 노력의 결과물이므로, 우리가 찾고 있는 관계를 서로 다른 방식으로 인코딩하고 서로 다른 관련 정보를 포함합니다.

각 데이터셋은 약간씩 다른 방식으로 처리되고 가져와지며, 하나의 그래프로 결합됩니다. 그림 C.5는 두 개의 장난감 데이터셋 (toy datasets)을 사용하여 실제로 병합이 어떻게 이루어지는지를 보여 줍니다.

![](images/d72b3e9eff4567cf9d912be6e59a32d8aeb2c785867b9940c376a007efde2504.jpg)  
그림 C.5 데이터셋 A는 더 균형 잡혀 있으며 참고 문헌을 포함합니다. 데이터셋 B는 더 적은 수의 질병에 초점을 맞추지만 더 많은 수의 miRNA를 포함합니다. 적재 후 두 출처의 정보는 단일 진실 공급원 (single source of truth)으로 통합됩니다.

![](images/7cfbbba6db04ac23fdc254aa9016313de76f3f1cc103a9733a718c5749dc1f3f.jpg)  
그림 C.6 첫 번째 반복의 대상 스키마. 여기에는 우리가 예측하고자 하는 관계 유형인 —REGULATES 및 RELATED\_TO—가 포함되어 있습니다.

두 출처, 이들이 포함하는 정보, 그리고 우리의 목표를 고려하면, 그림 C.6에 제시된 스키마가 현재 우리의 필요에 적합합니다. 여기서는 두 가지 유형의 관계를 사용하며, 이는 우리가 가져올 두 데이터셋 각각에 하나씩 대응합니다. 이러한 선택은 두 데이터셋이 관계에 부여하는 의미상의 뉘앙스를 보존할 수 있게 해 줍니다. 또한 쿼리 결과를 볼 때 특정 관계가 두 출처 중 어느 쪽에서 왔는지 빠르게 식별할 수 있습니다. 나중에 모델을 구축할 때 필요하다면 이 두 유형의 관계를 쉽게 병합할 수 있습니다.

우리는 HMDD 데이터셋을 가져오는 것부터 시작합니다. 이 데이터셋은 인간 miRNA–질병 연관-

관계에 대한 실험 기반 증거를 포함합니다. 이 데이터셋은 수작업으로 큐레이션 (curation)되었으며 우리가 예측하고자 하는 정확한 miRNA-질병 링크를 포착하므로, 우리의 KG를 위한 견고한 기반을 제공할 것입니다. 다음 목록은 데이터 적재 과정을 처리하는 HMDD 임포터 클래스의 구현을 보여 줍니다.

![](images/853d14f8e1fa920010249bfa9bf418091d8c9719049ab5c05390368a668c9c2b.jpg)

```python
SET r.description = item.description, Merges a Reference node
r.pmid=item.pmid, r.category = item.category using the PubMed ID as
MERGE (ref:Reference {pubmed_id:item.pmid}) identifier and connects
MERGE (m)-[:HAS_REFERENCE]->(ref) < it to the miRNA node
"IIII
Computes size = self.get_csv_size(HMDD_file, encoding="latin-1").
the number self.batch_store(query, self.get_rows(HMDD_file),
of records size=size, strategy="aggregate") 4 Aggregates and
available stores the records
def set_constraints(self):
with self._driver.session(database=self._database) as session:
Enforces > query = """
uniqueness for CREATE CONSTRAINT ON (a:Disease) ASSERT a.name IS UNIQUE;
Disease, miRNA, CREATE CONSTRAINT ON (a:MiRNA) ASSERT a.name IS UNIQUE;
Target, and
Reference nodes CREATE CONSTRAINT ON (a:Reference)
ASSERT a.pubmed_id IS UNIQUE;
CREATE CONSTRAINT ON (a:Target) ASSERT a.name IS UNIQUE"""
for q in query.split(";"): < Executes
try: the query
session.run(q)
except Neo4jClientError as e: Ignores errors
if (e.code !=
"Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists"):
raise e
[...]
if name == _main__':
importing = HMDDImporter(argv=sys.argv[1:])
importing.set_constraints()
importing.import_HMDD(HMDD_file)
importing.close()
```

다음 목록에는 HMDD 레코드에 대해 생성된 딕셔너리의 예가 포함되어 있습니다.

{'category': 'genetics\_GWAS',   
'mir': 'hsa-mir-502',   
'disease': 'Carcinoma, Renal Cell, Clear-Cell',   
'pmid': 27346408,   
'description': "Polymorphism at the miR-502 binding site in the 3'[...] "}

이 레코드는 miRNA hsa-mir-502가 신장암, 구체적으로 투명세포 신세포암과 연관되어 있으며, 이 연결이 PubMed ID 27346408로 식별되는 과학 출판물에 기술된 실험에 의해 뒷받침됨을 나타냅니다. 우리는 이 정보를

![](images/2af081d8c3502c8a9586b61ef466fd1b8376840929966db4af6a61b3718d0d57.jpg)  
그림 C.7 데이터베이스에 저장된 그래프 일부로서의 레코드

관련 miRNA 및 질병 노드를 생성하고 그림 C.7에 표시된 것처럼 RELATED\_TO 관계를 통해 이들을 연결함으로써 포착합니다.

이 가져오기 이후, 우리는 1,207개의 고유한 miRNA, 이들 miRNA 간의 18,732개의 고유한 연결, 그리고 849개의 고유한 질병을 수집했습니다. 이는 우리의 링크 예측 과제를 위한 견고한 기반을 제공합니다.

우리가 가져오는 두 번째 데이터셋은 dbDEMC입니다. 이는 인간 암에 초점을 맞춘 차등 발현 miRNA를 포함하는 통합 데이터베이스입니다. 여기에는 마이크로어레이 플랫폼과 miRNA 시퀀싱을 통해 얻은 403개의 miRNA 발현 데이터셋 모음이 포함되어 있습니다. 다음 목록은 이 데이터를 처리하는 dbDEMC 가져오기 도구의 구현을 보여줍니다.

### 목록 C.3 dbDEMC 데이터셋 가져오기


```python
class DBDEMCImporter(BaseImporter):
[...]
@staticmethod Defines how to
def get_rows(miRNA_file): < access the raw data
with open(miRNA_file, 'r+') as in_file:
reader = csv.reader(in_file, delimiter='\t')
header = next(reader)
for row in reader: Filters out any incomplete or
if len(row) < 2: < mistakenly parsed records
continue
record = dict(zip(header, row))
Keeps only
if record["Species"] != "Homo sapiens": ≤ human data
continue
if len(record["CancerSubtype"]) > 1: <
Selects the most
disease = record["CancerSubtype"].lower() specific description
else: available for the
disease = record["CancerType"].lower() current record
disease = (disease.replace(",", "")
.replace("/", " ")
.replace("-", " "))
Selects the
name = record["miRNA_ID"].lower().strip(). miRNA name
yield { < Generates a dictionary-encoded
"name": name, representation of the record
"disease": disease,
"experiment": record["ExperimentID"],
"regulated": record["Status"]
}
Defines how to ingest
def import_dbDEMC(self, miRDB_file): < data in the graph
exact_match_query = """
UNWIND $batch as item < Unwinds a batch of records
Merges the Disease MERGE (m:MiRNA {name: item.name}) < Merges the miRNA node using
node using the disease WITH m,item
name as identifier MERGE (n:Disease {name: item.disease})
```

SET n:DiseaseDbDEMC, n.name\_in\_db\_demc = item.disease   
> MERGE (m)-[r:REGULATES {regulated: item.regulated}]->(n) 관계를 병합합니다   
relationship SET r.source = 'dbDEMC', r.experiment = item.experiment   
IIII 사이에서 II IIII 개수를 계산합니다   
miRNA and the size = self.get\_csv\_size(miRDB\_file) self.batch\_store(exact\_match\_query, 수집할 레코드의 개수를 집계하고 저장합니다   
self.get\_rows(miRDB\_file), size=size) < 레코드

다음 목록은 이 데이터셋에 대해 생성된 레코드의 예를 포함합니다.

목록 C.4 dbDEMC 레코드 샘플   
{'name': 'hsa-miR-155',   
'disease': 'glioblastoma',   
'experiment': 'EXP00065',   
'regulated': 'UP'}

이 레코드는 실험 EXP00065가 건강한 성인 뇌 조직과 비교하여 교모세포종 종양에서 hsa-miR-155 miRNA 수준이 상승했음을 보여준다는 것을 나타냅니다(이 miRNA는 이러한 종양 세포에서 상향 조절됩니다).

이미 존재하는 경우 새 노드를 만들지 않는 MERGE 절을 사용함으로써, 우리는 지식 베이스 (knowledge base)의 풍부성을 향상합니다. HMDD 데이터셋을 통해 우리는 hsa-mir-155가 대조군과 비교하여 활성 다발성 경화증 병변에서 유의하게 과발현된다는 것을 알고 있습니다. 동일한 데이터셋은 또한 허혈성 심근병증(심장 근육에 영향을 미치는 질병)에서 hsa-miR-155 수준 증가가 관찰된다는 것도 나타냅니다. 두 번째 수집(dbDEMC로부터)이 끝나면, 그림 C.8은 이 특정 miRNA를 나타내는 동일한 노드가 종양성 및 비종양성 뇌 질환(각각 교모세포종과 다발성 경화증) 모두에 연결되어 있으며, 또한 뇌 및 심장 질환(각각 교모세포종과 허혈성 심근병증)에 연결되어 있음을 보여줍니다.

![](images/b8d5664d737f088341e3e4814e1cb4505917a265b0bfeb5a284445c8d9d729a7.jpg)  
그림 C.8 데이터셋을 병합하면 정보 융합이 가능합니다. 우리는 두 데이터베이스에 대해 두 가지 유형의 관계를 사용하고 있습니다. HMDD 데이터셋에는 RELATED\_TO를, dbDEMC에는 REGULATES를 사용합니다.

이제 세 가지 병리가 모두 hsa-miR-155의 상승한 수준에 의해 촉진되는 과도한 염증과 어떤 방식으로든 연결되어 있다고 가정해 보겠습니다. 또한 또 다른 새로운 병리가 동일한 염증 과정과 상관되어 있다고 가정해 보겠습니다. 이 경우 링크 예측 (link prediction) 모델은 hsa-miR-155 miRNA와 이 새로운 병리 사이의 아직 문서화되지 않은 관계를 추론하는 데 필요한 모든 정보를 갖습니다.

우리는 세 번째 데이터셋인 miR2Disease 데이터베이스를 사용하여 KG를 더욱 풍부하게 만들 수 있습니다. miR2Disease는 다양한 인간 질병에서 miRNA 조절 이상에 관한 정보를 제공하는 수작업 큐레이션 데이터베이스입니다. 또한 연구자들이 새로운 miRNA–질병 관계를 제출할 수 있도록 하는 제출 페이지를 제공하므로, 시간이 지남에 따라 성장할 것으로 기대할 수 있습니다. 다음 목록은 이 리소스를 우리의 KG에 통합하는 miR2Disease 가져오기 도구를 구현합니다.

![](images/bcd287c819e0807291cb034d8b4694bdcd5d525c13d7752b0dc1148dbcbe5c41.jpg)

이 세 가지 데이터셋을 가져오면, 우리의 KG는 4,874개의 고유한 miRNA, 이러한 miRNA 간의 118,806개의 고유한 연결, 그리고 1,144개의 고유한 질병으로 구성됩니다.

세 데이터셋 간에 고유한 miRNA가 어떻게 분포하는지 확인할 수 있습니다. 다음 목록은 이 분포를 계산하는 Cypher 쿼리를 보여주며, 이를 통해 각 데이터 소스의 중복과 고유한 기여를 분석할 수 있습니다.

### Listing C.6 수집된 miRNA 분포 계산


모든 miRNA 노드는 출처 데이터셋을 추적하기 위한 레이블을 가집니다.   
MATCH (n:MiRNA) < 출처 데이터셋을 추적합니다.   
WITH   
DISTINCT LABELS(n) AS labels, < miRNA 노드를 출처 데이터셋에 따라 그룹화합니다.   
COUNT(\*) as count 해당 노드가 어느 데이터셋에서 왔는지   
RETURN   
[l in labels where "MiRNA"<> l ] AS labels, ≤ “MiRNA”는 분포에 대한 정보를 제공하지 않으므로 레이블 목록에서 제거합니다.   
Count labels list because it provides no   
ORDER by count DESC information about distribution

쿼리의 결과는 다음과 같습니다.

```ini
[MiRNA_dbDEMC] 2550
[MiRNA_HMDD, MiRNA_dbDEMC] 583
[MiRNA_HMDD, MiRNA_dbDEMC, MiRNA_miR2Disease] 328
[MiRNA_HMDD] 280
[MiRNA_dbDEMC, MiRNA_miR2Disease] 84
[MiRNA_miR2Disease] 32
[MiRNA_HMDD, MiRNA_miR2Disease] 15
```

그림 C.9는 이러한 결과를 벤 다이어그램으로 보여줍니다. 공유 miRNA와 비공유 miRNA 사이에는 비교적 균형 잡힌 분포가 있습니다. 이는 공유 miRNA의 경우, 우리 KG와 이후의 ML 작업이 여러 데이터셋에서 파생된 지식의 이점을 얻을 수 있기 때문에 중요합니다. 비공유 miRNA는 우리가 수집한 각 데이터셋의 고유한 기여를 나타냅니다.

고유 miRNA 분포  
![](images/f458094765e159cf386f76544d9e3155db95f99fb153206880471fdcb7491e50.jpg)  
그림 C.9 miRNA의 고유 분포를 보여주는 벤 다이어그램입니다. 이 그림은 데이터셋 간에 중복되는 miRNA를 보여줍니다.

### C.2.2 질병 온톨로지 가져오기


가져온 데이터를 자세히 살펴보면, 데이터 소스들이 동일한 질병을 지칭하기 위해 서로 다른 용어를 사용한다는 점을 확인할 수 있습니다. 이는 흔한 문제로, 동일한 주제와 관련된 데이터셋이라 하더라도 객체를 정의할 때 서로 다른 표준을 사용하는 경우가 많기 때문입니다. 이는 생물학 및 의학 데이터셋의 경우 특히 자주 발생합니다. 이러한 불일치의 해로운 영향 중 하나는, 예를 들어 두 miRNA가 겉보기에는 서로 다른 두 질병에 연결되어 있을 수 있지만, 실제로는 서로 다른 이름으로 동일한 질병을 지칭할 수 있다는 점입니다.

그림 C.10은 이 문제의 예를 보여줍니다. 이 경우 각 데이터셋은 서로 다른 질병 명명 규칙에 의존하여 버킷 림프종을 서로 다른 표현으로 지칭했으며, 그 결과 세 개의 서로 다른 질병 노드가 생성되었습니다.

![](images/04945452c09f5e2069ec89dc300cab8da187ab423da4c1af1c7772498a9f651c.jpg)  
그림 C.10 이 miRNA들은 서로 다른 질병을 지칭하는 것처럼 보입니다.

그 결과, KG는 올바른 것으로 간주될 수 없습니다. 알고 있듯이, KG는 현실 세계 엔터티의 표현입니다. 서로 다른 엔터티가 동일한 개념을 나타낸다면, 그래프는 신뢰할 수 있는 진실의 원천으로서의 역할을 상실합니다. 더 나아가, 우리가 링크 예측 작업을 수행할 때 모델은 언급된 miRNA들이 서로 연결되어 있지 않다고 잘못 판단할 수 있으며, 이는 모델이 제대로 학습하는 것을 방해합니다. 일반적으로 이러한 유형의 불일치가 발생하면, 그 위에 구축하는 모든 표현은 그에 비례하여 저하됩니다.

### UMLS와 SCISPACY를 사용한 질병 정규화


다행히도 서로 다른 데이터셋의 질병 명명법을 “정규화”하는 데 사용할 수 있는 온톨로지 (ontology)가 많이 있습니다. 여기서는 통합 의학 언어 시스템 (Unified Medical Language System, UMLS; https://www.nlm.nih.gov/research/umls) [25] 온톨로지를 사용합니다.

UMLS는 전자 건강 기록을 포함하여 더 효과적이고 상호운용 가능한 생의학 정보 시스템과 서비스를 만들도록 촉진하기 위해, 관련 리소스와 함께 핵심 용어, 분류, 코딩 표준을 통합하고 배포합니다.

우리는 생의학, 과학 또는 임상 텍스트를 처리하기 위한 spaCy 모델을 포함하는 Python 패키지인 scispaCy (https://allenai.github.io/scispacy)를 사용할 것입니다. scispaCy는 UMLS 엔터티에 대한 자동 개체명 인식 (named entity recognition)을 수행할 수 있으며, 질병 이름 속성에서 식별된 모든 엔터티에 대해 정준명 (canonical name), 개념 ID, 유형 ID를 반환합니다. 이를 사용하여 우리가 수집한 모든 질병 노드의 정준명을 자동으로 추론하고, 동등한 Disease 노드에 연결할 새로운 Normalized-Disease 노드를 생성할 수 있습니다(그림 C.11 참조). 세 노드를 단일 노드로 병합하는 대신 스키마에 새 노드를 추가하기로 결정한 이유는 원래 구조를 유지하고자 하기 때문입니다. 이는 두 가지 이-

![](images/e350e29d150a6b1d513821c8e56c4904d820ea4b2c80b9260e011774079bfc9a.jpg)  
그림 C.11 NormalizedDisease 노드를 포함하는 대상 스키마 업데이트

유에서 유용합니다. 첫째, 결과를 검토하고 오류를 수정하기가 더 쉬워집니다. 둘째, 필요한 경우 모든 것을 재설정하고 다시 실행하는 일이 훨씬 더 간단해집니다.

다음 리스팅은 자연어 처리 (natural language processing, NLP) 기법을 사용하여 서로 다른 데이터 소스 전반의 질병 엔터티를 표준화하는 질병 정규화 과정을 구현합니다.

```python
Listing C.7 Normalizing diseases
class Reconciliator(BaseImporter): < Reuses functionality
def _init__(self, argv): from BaseImporter
super().__init__(command=__file__, argv=argv)
self._database = "hmdd2.0"
self.resolver = DiseaseResolver()
Defines how to access
def get_normalized_diseases(self): < the dataset's raw data
with self._driver.session(database=self._database) as session:
diseases_data = session.run("""
Fetches diseases to be MATCH (d:Disease)
normalized from Neo4j RETURN id(d) as id, d.name as name""").data()
Extracts the
diseases_text = [d["name"] for d in diseases_data]. < disease name
Converts itemslike “leukemia, disease_ids = [d["id"] for d in diseases_data] < Extract the
lymphocytic, chronic, Disease node ID
b-cell” into “b-cell diseases_text = [
chronic lymphocytic " ".join(i for i in reversed(d.split(","))).strip()
leukemia” for d in diseases_text]
```

```python
diseases_items = [self.resolver.nlp(disease)
for disease in diseases_text]
Runs the NLP pipeline for
every disease name text
disease_normalized = [ and returns an object
self.resolver.normalize(item) containing the detected
for item in diseases_items] < medical entities
Normalizes the disease
diseases = [{ using information from
"source_id": disease_id, the scispaCy pipeline
"name": disease_name,
"umnls_id": disease_UMNLS_ID} Converts normalized
for disease_id, (disease_name, disease_UMNLS_ID) diseases into a
in zip(disease_ids, disease_normalized)] < dictionary
return diseases
Defines how
def import_normalized_diseases(self): < to ingest data
query = """
Unwinds a UNWIND $batch as item Creates a node
batch of records MATCH (d:Disease) Selects the originalDisease node representing the normalized disease
<
MERGE (nd:NormalizedDisease {name:item.name}) < if it does not exist
SET nd.umnls_id = item.umnls_id
Links the original disease
MERGE (d)-[:REPRESENTS]->(nd) <
with its normalized version
"""
diseases = self.get_normalized_diseases()
Aggregates and self.batch_store(query, iter(diseases),
stores records size=len(diseases), strategy="aggregate") #O
```

다음 리스팅은 DiseaseResolver에 캡슐화된 해소 로직을 보여줍니다.

### 리스팅 C.8 DiseaseResolver 클래스

질병으로 간주되기 위해 완전히 일치해야 하는 엔터티 유형 집합을 정의합니다.   
class DiseaseResolver:   
full = ["Finding", "Organ or Tissue Function", "Tissue"] <   
banned = ["Human", "Body Part, Organ, or Organ Component",   
"Qualitative Concept", "Temporal Concept",   
"Functional Concept", "Body Space or Junction",   
"Spatial Concept"] < 유효한 질병으로 간주될 수 없는 엔터티 유형 집합을 정의합니다.   
def init\_\_(self):   
self.nlp = nlp = spacy.load("en\_core\_sci\_sm") scispaCy   
config = { NLP 모델을 생성합니다.   
"resolve\_abbreviations": True,   
"linker\_name": "umls"}   
링커를 가져옵니다. nlp.add\_pipe("scispacy\_linker", config=config) < UMLS 엔터티를 감지하는 링커를 설정합니다.   
the linker linker = nlp.get\_pipe("scispacy\_linker")   
self.type\_tree = linker.kb.semantic\_type\_tree. <   
self.cui\_to\_entity = linker.kb.cui\_to\_entity   
UMLS 엔터티에서 UMLS 온톨로지가 색인으로 사용하는 개념 ID로의 매퍼를 가져옵니다. 엔터티에 레이블을 지정하기 위해 UMLS에서 유형의 의미 트리를 가져옵니다(예: "multiple sclerosis"는 “Disease or Syndrome”으로 레이블이 지정됩니다).

```python
def canonical(self, entity): ≤ Gets the assigned
"""get canonical name from entity""" canonical name
entities = entity._.kb_ents for an entity
if len(entities) == 0:
return
### select the first entity
return self.cui_to_entity[entities[0][0]].canonical_name
def types(self, entity): < Gets the types associated
"""return semantic types for the entity""" with an entity
entities = entity._.kb_ents
if len(entities) == 0:
return []
return [self.type_tree.get_canonical_name(t)
for t in self.cui_to_entity[entities[0][0]].types]
@staticmethod Checks whether the entity
def matchesAll(entity): < spans the entire text
"""return true if the entity covers the whole content"""
return entity.start == 0 and entity.end == len(entity.doc)
def containsOnly(self, entity, targets): < Checks whether an entity contains
"""return true if the entity types are only a specific set of type labels
within the target types"""
intersection = set(self.types(entity)).intersection(targets)
return (intersection == set(self.types(entity)))
Checks whether an entity
def validEntity(self, entity): < can be considered a disease
""" exploits the entity types to detect if an entity is
correctly identified as disease """
If the type V if self.containsOnly(entity, self.banned):
If the type labels are among those
labels are return False
defined earlier which can represent
among those if self.containsOnly(entity, self.full): a disease only if the entity spans
defined earlier return self.matchesAll(entity) < the entire text, and that is not the
that never return True < The type labels case, the model failed to recognize
apply to represent valid diseases. the disease as a whole.
disease, the def normalize(self, item):
model failed """"main entrypoint: convert item into a normalized disease
to recognize return ( normalized_name, UMNLS_ID None )
the disease. II IIII If only a single entity is found,
if len(item.ents) == 1: we use the normalization logi
If no entities return self.normalize_entity(item) < defined in normalize_entity.
are found, we if len(item.ents) > 1:
use the default return self.normalize_default(item) ≤ If more than one entity is found
normalization. > return self.normalize_default(item)
for the current disease, we use
the default normalization.
def normalize_entity(self, item):
""" normalize item when there is only one detected entity """
Defines the logic entity = item.ents[0]
to normalize a disease if only if self.validEntity(entity): return self.canonical(entity), entity._.kb_ents[0][0] <
one entity is found return self.normalize_default(item) If the detected entity
is a valid disease, we return the tuple
If the detected entity is not a valid disease,
(<canonical name>, <UMLS Concept Id>)
we use the default normalization.
extracted from the entity metadata.
```

```python
def normalize_default(self, item): < Defines the default logic
"""When no other better options are available to normalize a disease
return capitalized version of disease"""
item = str(item)
item = " ".join(i.capitalize() for i in item.split())
return item.strip(), None < Returns a capitalized version of
the disease text and None as ID
```

이 정규화 단계 이후, 철자가 서로 다른 세 개의 Burkitt 림프종 노드는 하나의 정규화된 Burkitt Lymphoma 노드로 연결됩니다(그림 C.12 참조). 우리는 이전에는 연결되지 않았던 구성요소들을 연결하는 추가 노드를 더함으로써 그래프 단편화를 효과적으로 줄였습니다. 책의 다른 곳에서 사용하는 약연결 구성요소 (weakly connected component, WCC) 알고리즘을 사용하여 정규화 노드가 있을 때와 없을 때의 그래프 연결성을 평가할 수 있습니다. WCC를 사용하면 하나의 연결 구성요소를 형성하는 연결된 노드 집합을 감지할 수 있습니다. 다시 말해, 그래프에서 연결되지 않은 하위 그래프의 수를 식별하고, 각 노드가 속한 연결되지 않은 하위 그래프에 따라 각 노드에 레이블을 지정할 수 있습니다.

![](images/3cb75aebd21d471cc0aaac6f14d7e3410c7f23b2caf29edbd88050863e791085.jpg)  
그림 C.12 이제 miRNA는 정규화된 질병 노드인 Burkitt Lymphoma에 연결됩니다.

### 정규화 효과 평가


우리의 정규화를 통해 이전에는 연결되지 않았던 그래프들을 연결했을 수 있습니다. 정규화 과정 전후에 WCC를 실행하고 결과를 비교함으로써 이를 측정할 수 있습니다.

WCC를 실행하기 전에 다른 모든 GDS 알고리즘과 마찬가지로, 분석하려는 그래프의 명명된 인메모리 표현을 생성해야 합니다. 이 경우 두 가지 표현을 생성합니다. 하나는 NormalizedDisease 노드와 그 관련 관계를 포함하는 표현이고, 다른 하나는 이를 포함하지 않는 표현입니다. 다음 목록은 이 두 그래프 표현을 메모리에 투영하는 Cypher 쿼리를 보여 주며, 비교 분석을 위한 환경을 설정합니다.

![](images/38b7486082273d84b5509c20a9382788c7435a58809e8895ad12d7cdbd947ffe.jpg)

이제 두 인메모리 표현 모두에 대해 WCC 알고리즘을 한 번씩 실행하고 결과를 비교할 수 있습니다. 다음 목록은 정규화되지 않은 그래프에서 WCC 알고리즘을 실행합니다.

![](images/35231c1bc9cb71a18b1c44c0b5c04cb8fa2b4e591d73b31d6d4b301512ed65e3.jpg)  
표 C.1은 정규화 전 구성요소의 분포를 보여 주며, 질병 엔터티 표준화 이후와 비교할 기준선 연결성 패턴을 드러냅니다. 정규화되지 않은 그래프의 연결성을 검토한 후에는, 질병 표준화의 영향을 평가하기 위해 정규화된 표현에 대해 동일한 분석을 수행할 수 있습니다(목록 C.11 참조).

표 C.1 정규화 전 WCC 구성요소 분포
<table><tr><td>하위 그래프</td><td>구성요소 크기</td></tr><tr><td>0</td><td>5010</td></tr><tr><td>1166</td><td>3</td></tr><tr><td>1838</td><td>2</td></tr></table>

목록 C.11 정규화된 그래프에서 WCC 실행   
CALL gds.wcc.stream('normalized') 각 노드의 ID와   
YIELD nodeId,componentId < 해당 노드가 속한 하위 그래프의 ID를 반환합니다   
RETURN componentId AS Subgraph, count(nodeId) AS ComponentSize <   
WCC 알고리즘을 호출하고 인메모리 표현을 수정하지 않고 결과를 반환합니다 각 구성요소(하위 그래프)의 크기 분포를 계산합니다

표 C.2는 정규화 후 구성요소 분포를 보여 주며, 그래프 응집성의 개선 정도를 정량화하기 위해 정규화 이전 상태와 직접 비교할 수 있게 합니다. 이 경우 정규화 전후의 그래프 구조를 비교할 때 유의미한 변화는 없습니다. 거의 모든 노드를 포함하는 하나의 크고 연결된 구성요소만 있기 때문입니다.

표 C.2 정규화 후 WCC 구성요소 분포
<table><tr><td>하위 그래프</td><td>구성요소 크기</td></tr><tr><td>0</td><td>6033</td></tr><tr><td>1166</td><td>4</td></tr><tr><td>1838</td><td>3</td></tr></table>

무시할 수 있을 정도로 단편화가 적고 잘 연결된 그래프를 갖는 것은 대부분의 그래프 애플리케이션에 좋은 소식이지만, 여기서는 정규화 단계의 영향을 정량화하기 위해 다른 기법을 적용해야 합니다. GDS 라이브러리에는 정규화 과정으로 인한 구조적 변화를 통계적으로 평가할 수 있는 여러 커뮤니티 탐지 알고리즘이 포함되어 있습니다. 그러나 WCC를 사용할 때보다 다른 커뮤니티 탐지 알고리즘을 사용할 때 이러한 변화는 해석하기가 더 어렵습니다.

그림 C.13에 표시된 하위 그래프를 고려해 보겠습니다. 여기서 hsa-mir-199a\*는 간세포암종 (hepatocellular carcinoma, hcc)을 통해 hsa-mir-182에 연결되며, 동시에 hsa-mir-182는 유두상 갑상샘암종 (carcinoma, papillary, thyroid)을 통해 hsa-mir-4728에 연결됩니다. 두 miRNA를 연결하는 최단 경로에는 Disease 노드가 하나만 포함되므로, hsa-mir-199a\*와 hsa-mir-182 사이의 거리는 1과 같다고 말할 수 있습니다. hsa-mir-199a\*와 hsa-mir-4728 사이의 거리는 2와 같은데, 이는 최단 경로가 두 개의 Disease 노드를 통과하기 때문입니다. 그러나 이제 우리는 burkitt’s lymphoma와 lymphoma, burkitt가 실제로 동일한 질병임을 알고 있습니다. 따라서 hsa-mir-199a\*와 hsamir-4728 사이의 거리는 1이어야 합니다.

그림 C.14는 질병 정규화 이후 동일한 miRNA 사슬을 보여 줍니다. 이전에는 서로 분리되어 있던 질병 노드들이 이제 공유된 NormalizedDisease 노드를 통해 연결되어, 관련 miRNA 사이의 경로 길이가 효과적으로 줄어듭니다.

![](images/224bc4248ba518800993f7550bfedd863d90300f2b67f4ba8f4419bd6bc323d6.jpg)  
그림 C.13 Disease 노드를 통해 연결된 miRNA 사슬

![](images/8c6fe7c2e1499362e5ace63182c96b262613d809386f8d0d6ae9d6d75fcbceaf.jpg)  
그림 C.14 NormalizedDisease 노드를 통한 연결 이후, 그림 C.13과 동일한 miRNA 사슬

일반적으로 우리는 질병 정규화 이후 miRNA 사이의 거리가 더 짧아질 것으로 예상합니다. 이러한 거리는 모든 쌍 최단 경로 (all-pairs shortest path, APSP) 알고리즘을 사용하여 측정할 수 있으며, 이 글을 쓰는 시점에는 Neo4j GDS 라이브러리의 경로 탐색 알고리즘에서 알파 등급으로 제공됩니다.

먼저 두 miRNA 노드가 모두에 연결된 Disease 노드가 하나 이상 존재할 때에만 서로 연결되는 인메모리 그래프 표현을 생성하겠습니다. 이 프로젝션은 정규화 단계 이전의 그래프 상태를 나타냅니다.

또한 두 miRNA의 노드가 이를 연결하는 (Disease)-[]-(NormalizedDisease)-[]- (Disease) 사슬이 존재할 때 연결되는 두 번째 인메모리 그래프 표현도 생성하겠습니다. 이 프로젝션은 정규화 단계 이후의 그래프 상태를 나타냅니다. 다음 Cypher 질의는 공유된 Disease 노드를 통해 miRNA를 연결하는 이 인메모리 그래프를 생성합니다.

![](images/e87745eda8be17665b9064d66de68bfe051ed5a887667a1c4412c49320e48c5e.jpg)

다음 목록은 질병을 정규화하기 위해 생성된 NormalizedDisease 노드를 사용하는 인메모리 그래프를 생성합니다.

리스팅 C.13 NormalizedDisease 노드를 통한 miRNA-대-miRNA 연결   
call gds.graph.project.cypher(  그래프를 생성합니다   
"NormalizedDiseaseDistance", 그래프의 이름을 지정합니다   
Selects all D "MATCH (n:MiRNA) return id(n) as id",   
miRNA "MATCH p1=(a:MiRNA)-[:REGULATES|RELATED\_TO]->()-[:REPRESENTS]->(d)   
노드 MATCH p2=(d)<-[:REPRESENTS]-()<-[:REGULATES|RELATED\_TO]-(b:MiRNA) <   
WHERE id(a)<id(b) < 다음 경우 b -> a를 무시합니다   
RETURN distinct #F a -> b가 이미 존재합니다. 다음을 생성합니다   
id(a) as source, 관계:   
id(b) as target") < a와 b가 연결되어 있으므로 노드 ID를 반환합니다   
a와 b는 여러 NormalizedDisease를 통해 연결될 수 있지만, 하나를 통해 연결되어 있는 경우 이를 생성하기 위해서는 출발지와 목적지만 필요합니다

이 쿼리들은 실행하는 데 어느 정도 시간이 필요할 수 있습니다. 일반적으로 gds.graph.project 호출은 gds.graph.project.cypher 호출보다 더 빠른데, 전자는 데이터베이스에 이미 저장되어 있는 그래프 노드와 엣지에 대한 정보를 사용하기 때문입니다. 반면 Cypher 투영 (Cypher projection)은 더 유연하며, 방금 수행한 것과 유사하게 계산을 사용하여 한 그래프를 다른 그래프 위에 투영할 수 있으므로 탐색 및 디버깅 목적에 더 유용합니다.

리스팅 C.14는 정규화 전 거리를 계산하고, 리스팅 C.15는 정규화 후 거리를 계산합니다. 두 리스팅의 결과는 표 C.3에 요약되어 있습니다.

리스팅 C.14 정규화 전 그래프에서 APSP 실행   
CALL gds.allShortestPaths.stream('DiseaseDistance',{}) <   
YIELD distance   
무방향 APSP를 실행합니다   
RETURN distinct distance, count(distance) AS Count 그리고 결과를 반환합니다   
메모리 내 표현을 수정하지 않고 거리를 반환합니다   
분포

리스팅 C.15 정규화 후 그래프에서 APSP 실행   
CALL gds.allShortestPaths.stream('NormalizedDiseaseDistance',{}) <   
YIELD distance   
무방향 APSP를 실행합니다   
RETURN distinct distance, count(distance) AS Count #B 그리고 결과를 반환합니다   
메모리 내 표현을 수정하지 않고 거리를 반환합니다   
분포

표 C.3 거리 분포
<table><tr><td rowspan=1 colspan=1>거리</td><td rowspan=1 colspan=1>이전 경로 수</td><td rowspan=1 colspan=5>이후 경로 수</td><td rowspan=1 colspan=1>변화</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5,911,305</td><td rowspan=2 colspan=5>6,179,3291,010,851</td><td rowspan=2 colspan=1>+4.5%-18.7%</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1,244,305</td></tr><tr><td rowspan=2 colspan=1>34</td><td rowspan=1 colspan=1>35,795</td><td rowspan=1 colspan=2>25,888</td><td rowspan=2 colspan=3></td><td rowspan=2 colspan=1>-27.6%-29.6%</td></tr><tr><td rowspan=1 colspan=1>870</td><td rowspan=1 colspan=5>612</td></tr><tr><td rowspan=2 colspan=1>5</td><td rowspan=2 colspan=1>46</td><td rowspan=2 colspan=4>22</td><td rowspan=1 colspan=1></td><td rowspan=3 colspan=1>-52.1%-100%</td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=5>0</td></tr></table>

거리 1에 있는 miRNA 쌍의 수가 약 5% 증가했음을 확인할 수 있으며, 이는 이러한 쌍의 거리가 짧아졌다는 의미입니다. 다른 거리에서의 수가 감소한 것 역시 더 짧은 거리로의 이동을 시사합니다.

이는 중요한 분석이자 주목할 만한 성과입니다. 많은 임베딩 기법은 메시지 전달 (message passing)에 기반하는데, 이는 각 반복에서 새로운 임베딩을 계산하는 데 사용되는 메시지를 전달하기 위해 관계를 활용한다는 의미입니다. 관련 노드들 사이의 더 짧은 연결은 더 높은 품질의 최종 임베딩과 직접적으로 관련됩니다.

### 엔티티 정규화에 LLM 사용하기


scispaCy를 사용한 우리의 접근법은 burkitt lymphoma, lymphoma, Burkitt, Burkitt’s lymphoma와 같은 질병 엔티티를 성공적으로 정규화했지만, 전문 생의학 NLP 도구를 사용할 수 없거나 적합하지 않은 상황에서는 LLM이 대안적 해결책을 제공할 수 있습니다. LLM은 사전학습 중 방대한 생의학 문헌 말뭉치에 노출된다는 이점을 가지며, 이를 통해 용어 변이에 대한 내재적 이해를 갖게 됩니다.

LLM은 여러 방식으로 엔티티 조정 (entity reconciliation) 문제를 해결할 가능성이 있습니다.

용어 변이 처리—LLM은 용어가 예측 가능한 변환 패턴을 따르지 않더라도 의미적 동등성을 인식할 수 있습니다. 예를 들어, 서로 다른 단어를 사용하고 있음에도 “Gastric adenocarcinoma”와 “Stomach cancer”가 동일한 엔티티를 가리킨다는 것을 인식할 수 있습니다.

도메인 불문 적용—여기서 제시한 엔티티 조정 접근법은 생의학 도메인에 국한되지 않습니다. LLM은 법률 문서, 재무 보고서, 기술 명세서처럼 scispaCy와 같은 전문 도구를 사용할 수 없는 다양한 도메인에서도 유사한 정규화 기능을 제공할 수 있습니다.

제로샷 (zero-shot) 능력—UMLS와 같은 특정 온톨로지 (ontology)를 필요로 하는 우리의 현재 접근법과 달리, LLM은 특히 일반적인 엔티티의 경우 외부 지식 베이스 없이도 엔티티 정규화를 합리적으로 잘 수행할 수 있을 것입니다.

이러한 잠재적 이점에도 불구하고, 핵심적인 한계를 고려해야 합니다. LLM 출력의 확률적 특성은 실행마다 일관되지 않은 엔티티 매핑을 초래할 수 있으며, 이는 KG 구축 과정의 재현성을 저해할 가능성이 있습니다. 또한 대규모 엔티티 조정에 LLM을 배포하려면 scispaCy와 같은 더 경량의 접근법에 비해 상당한 컴퓨팅 자원이 필요합니다.

### C.2.3 miRNA 정보 가져오기

지금까지 우리는 알려진 miRNA-질병 연관성을 포함하는 데이터셋을 수집하고, 질병 정규화 및 병합을 통해 질병 관계의 품질을 개선했습니다. 두 단계 모두 풍부하고 고품질의 관계를 구축하는 데 도움이 되며, 링크 예측 과제를 위한 견고한 기반을 제공합니다. miRNA와 그 연결에 관한 정보를 제공하는 추가 데이터셋을 수집함으로써 KG를 더욱 풍부하게 만들 수 있습니다.

그림 C.15의 업데이트된 스키마에서 보이듯이, 새로운 관계 중 일부는 miRNA들을 서로 연결하여 miRNA 유사성과 관계에 대한 직접적인 정보를 제공합니다(직접 유사성). miRNA와 표적(Target), miRNA와 참조(Reference)와 같은 다른 관계들은 간접적인 연결 정보를 제공합니다(간접 유사성). 예를 들어 두 miRNA가 동일한 표적 mRNA에 결합한다면, 이들은 동일한 유전자 발현을 조절하거나—또는 침묵시킨다는—의미에서 유사합니다. 마찬가지로, 동일한 출판물에서 인용된 두 개 이상의 miRNA는 적어도 저자들의 관점에서는 어떤 방식으로든 관련이 있음을 나타냅니다. 그림 C.16은 예시 데이터셋을 통해 이 아이디어를 설명합니다.

![](images/ada04a0fb623f4b6b70b206ffac2008e960c329b563db3d945e78242ff9c6d3e.jpg)  
그림 C.16 직접 유사성과 공유 노드에 의해 유도된 유사성의 예입니다. 모델은 링크 예측 과제에 가장 관련성이 높은 유사성이 무엇인지 학습하게 됩니다.

miRBase 데이터셋(www.mirbase.org) [26–29]을 가져오는 것부터 시작하겠습니다. 이는 약 200개의 출판된 miRNA 서열과 주석을 검색할 수 있는 데이터베이스입니다. 각 miRNA에 대해 이 데이터셋은 해당 miRNA를 언급하는 관련 출판물 목록과 연결된 miRNA 목록을 보고합니다. 다음 목록은 이 miRNA 참조 데이터베이스를 우리 KG로 추출하고 통합하는 작업을 처리하는 miRBase 가져오기 도구를 구현합니다.

![](images/df7bc7ea33521376e25c77a19841ce06f059496a8b92c09100592ec85ccef0ba.jpg)

for record in SeqIO.parse(miRNA\_dat, "embl"): <   
데이터셋을 읽습니다.   
if not record.name.startswith("hsa"): EMBL에 저장됩니다.   
continue 형식이며, 이는   
if len(record.name) < 2: 중첩 구조를 포함합니다.   
continue   
yield { 비어 있거나   
"name": record.name.lower(), 인간과 관련 없는 miRNA를 필터링합니다.   
"description": record.description,   
"seq": str(record.seq),   
"comment": record.annotations.get('comment', ''),   
"references": [   
{"authors": r.authors, "title": r.title,   
"pubmed\_id": r.pubmed\_id, "journal": r.journal}   
for r in (record.annotations   
.get('references', []))], 레코드의   
"features": [   
출판 참고문헌을 추출합니다.   
{"type": r.type,   
"accession": (r.qualifiers   
.get('accession', [""])[0]),   
"name": (r.qualifiers   
현재 항목과 관련된 .get('product', [""])[0]   
miRNA를 추출합니다 .lower())}   
the current one > for r in record.features if r.type == "miRNA"]   
}   
def import\_miRNA\_dat(self, miRDB\_file):   
query = """ 배치를 펼칩니다.   
UNWIND \$batch as item < 레코드의   
MATCH (m:MiRNA {name: item.name}) <   
현재 레코드에   
해당하는 miRNA 노드를 선택합니다.   
Updates the node m:MiRNA\_miRBase, current record   
이 데이터셋의 정보로 with this dataset’s m.description = item.description,   
m.seq = item.seq, m.comment = item.comment 각 feature를 반복적으로 처리합니다.   
WITH m,item item.features 목록에서   
현재 miRNA와 feature 목록의 항목 사이에 관계를 생성합니다. FOREACH (feature in item.features <   
MERGE (f:MiRNA {name: feature.name}) < feature item으로 식별되는   
miRNA 노드를 선택합니다.   
MERGE (m)-[:HAS\_FEATURE]->(f)   
)   
WITH m,item   
각 출판 참고문헌을 반복적으로 처리합니다. > UNWIND item.references as reference   
item.references 목록에서 MERGE (r:Reference {pubmed\_id: reference.pubmed\_id}) <   
ON CREATE SET r.authors = reference.authors, 생성합니다.   
r.title = reference.title,   
출판   
현재 r.journal = reference.journal Reference 노드가   
miRNA 노드를 > MERGE (m)-[:HAS\_REFERENCE]->(r) 존재하지 않으면 생성합니다.   
출판 Reference 노드와 연결합니다. IIII I   
size = self.get\_embl\_size(miRDB\_file) <   
D self.batch\_store(query, self.get\_rows(miRDB\_file), size=size)   
레코드를 집계하고 저장합니다. 수집할 레코드 수를 계산합니다.

다음 목록은 miRBase가 제공하는 레코드의 예입니다. 이는 miRNA를 서로 연결하고 참고문헌 논문에 연결할 수 있게 해 주는 풍부한 관련 정보를 포함합니다.

Listing C.18 신뢰도 점수를 포함한 miRDB 레코드 예시   
{'name': 'hsa-mir-96-5p',   
'target': 'NM\_012214',   
'value': 90.3926}

Listing C.17 miRBase 레코드 예시   
name: hsa-let-7a-1   
description: Homo sapiens let-7a-1 stem-loop   
comment:   
[6]에서 클로닝된 let-7a-3p는 1 nt 3' extension (U)을 가지며, 이는   
유전체 서열과 호환되지 않습니다.   
seq:   
UGGGAUGAGGUAGUAGGUUGUAUAGUUUUAGGGUCACACCCACCACUGGGAGAUAACU   
AUACAAUCUACUGUCUUUCCUA   
features:   
- accession: MIMAT0000062   
name: hsa-let-7a-5p   
type: miRNA   
accession: MIMAT0004481   
name: hsa-let-7a-3p   
type: miRNA   
references:   
- authors: Lagos-Quintana M, Rauhut R, Lendeckel W, Tuschl T   
journal: Science. 294:853-858(2001).   
pubmed\_id: 11679670   
title:   
소형 발현 RNA를 코딩하는 새로운 유전자의 식별   
authors:   
Suh MR, Lee Y, Kim JY, Kim SK, Moon SH, Lee JY, Cha KY,   
Chung HM, Yoon HS, Moon SY, Kim VN, Kim KS   
journal: Dev Biol. 270:488-498(2004).   
pubmed\_id: 15183728   
title:   
인간 배아 줄기세포는 고유한 miRNA 집합을 발현합니다.   
[...]

다음으로 수집할 데이터셋은 miRNA 표적 예측 및 기능 주석을 위한 온라인 데이터베이스인 miRDB (https://mirdb.org) [30, 31]입니다. 앞서 논의했듯이, miRNA는 주로 표적 유전자의 발현을 하향 조절함으로써 기능합니다. 따라서 miRNA 표적을 정확히 예측하는 것은 miRNA 기능을 특성화하는 데 매우 중요합니다. 이 데이터셋의 표적은 고처리량 시퀀싱 실험에서 얻은 miRNA–표적 상호작용을 분석하여 개발된 생물정보학 도구인 miR-TargetLink 2.0 (https://ccb-compute.cs.uni-saarland.de/mirtargetlink2) [32]을 사용해 예측됩니다. miRDB에는 5개 종에서 7,000개의 miRNA에 의해 조절되는 350만 개의 예측 표적이 포함되어 있습니다. 그러나 우리는 이미 그래프에 가져온 miRNA, 즉 인간과 관련된 miRNA에 초점을 맞출 것입니다.

다음 목록은 miRNA–표적 연관성의 강도를 평가하는 데 도움이 되는 신뢰도 점수를 포함한 miRDB 레코드 샘플을 보여줍니다. 가져오기 결과는 그림 C.17에 제시되어 있습니다.

![](images/f74d83d55026f04a240806d61825fdaf87c31fb7936ae41f5312d828a8be1a96.jpg)

마지막 단계로, miRNA 쌍별 기능적 유사성을 포함하는 비교적 작은 데이터셋을 가져올 것입니다. 유사도 점수는 생물정보학 도구인 MISIM (http://www.lirmed.com/misim) [33]을 사용해 얻어지며, 이 도구는 두 miRNA와 관련된 질병의 의미론적 값을 비교하여 miRNA 기능적 유사성을 계산합니다.

다음 MISIM 레코드 샘플에도 신뢰도 점수가 포함되어 있습니다. 가져오기 결과는 그림 C.18에 표시된 작은 그래프와 같습니다.

![](images/d23760af1f09931220f9441ff20391e16a3d69cd166f52325d6687f4c6a67b4f.jpg)

이 마지막 가져오기로 우리의 수집 과정이 완료됩니다. 다음으로 넘어가기 전에, 데이터베이스의 내용을 확인하고, 데이터베이스에 익숙해지며, KG에서 서로 다른 구성 요소가 어떻게 상호작용하는지 살펴보기 위해 다음의 간단한 연습을 실행해 보는 것이 유용할 수 있습니다. 연습 문제의 질문들은 데이터베이스의 규모를 파악하는 데 유용합니다. 이는 특정 알고리즘을 실행하는 데 필요한 시간과 생성된 모델의 최종 품질에 영향을 미치므로, 더 심층적인 분석을 수행하기 전에 이를 고려하는 것이 일반적으로 좋은 관행입니다.

데이터베이스에는 각 유형의 노드가 몇 개씩 존재합니까? 즉, miRNA는 몇 개이며, 질병은 몇 개입니까?

 어떤 질병이 가장 많은 miRNA와 연결되어 있습니까? 중앙값은 얼마입니까?

어떤 miRNA가 서로 다른 질병과 더 많은 연결을 가지고 있습니까? 중앙값은 얼마입니까?

### C.3 miRNA KG 탐색 및 분석


더 복잡한 작업으로 넘어가기 전에, 우리가 구축한 그래프를 검토하고 일부 정보를 추출하며 그 안에 포함된 지식의 품질을 검증해 보겠습니다. 이를 위해 동일한 유형의 노드들 사이 및 서로 다른 유형의 노드들 사이의 유사성을 관찰할 수 있게 해주는 쿼리를 실행할 수 있습니다. 이러한 종류의 유사성은 ML 알고리즘이 자신의 작업을 수행하는 데 사용할 암묵적 관계를 나타냅니다. 앞으로 살펴보겠지만, 학습 단계에서 많은 임베딩 알고리즘은 원하는 결과를 얻는 데 어떤 암묵적 관계가 더 유용한지를 식별하는 방법을 학습합니다.

데이터베이스를 생성하지 않은 경우, 우리의 데이터베이스(https://downloads .graphaware.com/neo4j-db-seeds/hmdd2.0.backup)를 가져올 수 있습니다. 전체 프로세스를 실행하는 경우 이 백업을 사용하여 데이터베이스를 검증할 수도 있습니다. neo4j.conf 파일에 다음 줄을 추가합니다.

dbms.databases.seed\_from\_uri\_providers=URLConnectionSeedProvider

그런 다음 다음 명령을 실행합니다.

목록 C.20 백업에서 miRNA 데이터베이스 가져오기

CREATE DATABASE \`hmdd2.0\` OPTIONS {existingData: "use", seedUri: "https://downloads.graphaware.com/neo4j-db-seeds/hmdd2.0.backup"}

예를 들어, miRNA들이 공통으로 가지는 표적 mRNA 수를 기반으로 서로 얼마나 유사할 수 있는지 평가해 보겠습니다. 두 miRNA가 많은 Target 노드를 공유한다면 이들을 유사하다고 간주하겠습니다. 그림 C.19는 이 개념을 보여 주며, miRNA2가 miRNA1보다 miRNA3에 더 유사하다는 점을 나타냅니다. 여기서 화살표 선의 두께는 표적 연결의 강도를 나타냅니다.

![](images/9bd1e7ac393c5b77e61b012e18f23f5704c93b4d839b97dd2fc23d7d105f19b4.jpg)  
그림 C.19 miRNA2는 miRNA1보다 miRNA3에 더 유사합니다. 화살표 선이 두꺼울수록 표적의 연결이 더 강합니다.

이러한 유형의 유사성을 계산하기 위해 GDS의 nodeSimilarity 함수를 가중 버전으로 사용하겠습니다. 이는 높은 점수 값을 가진 표적에 연결된 miRNA가 더 약한 연결을 가진 miRNA보다 더 중요하게 간주됨을 의미합니다. 이 알고리즘을 사용하기 전에, 분석할 그래프의 명명된 인메모리 표현을 생성해야 합니다.

목록 C.21 인메모리 그래프 생성   
CALL gds.graph.project("MiRNA\_Target\_similarity", Target과 MiRNA만 고려함   
["Target","MiRNA"], < miRNA와 Target   
{HAS\_TARGET:{properties:["value"]}}) <   
나중에 사용할 수 있도록 value 속성을   
인메모리 투영에 포함함

인메모리 데이터베이스가 생성되면 노드 유사도 계산을 실행할 수 있습니다. 다음 목록은 알고리즘의 가중 버전을 사용하여 노드 간 유사도를 계산하며, 이를 통해 더 높은 점수 값을 가진 연결을 우선시할 수 있습니다.

### 목록 C.22 유사도 계산


```sql
CALL gds.nodeSimilarity.stream(
"MiRNA_Target_similarity", Uses the value relationship
{relationshipWeightProperty: 'value'}) ≤ attribute as a weight property
YIELD node1,node2, similarity
WITH gds.util.asNode(node1) AS source,
gds.util.asNode(node2) AS target, similarity
RETURN source.name AS source, target.name AS target, similarity
ORDER BY similarity DESC, source, target
```

노드 유사도 계산 결과는 표 C.4에 보고되어 있습니다. 표에서 가장 유사한 miRNA들에 대해, 우리는 명백한 예(예: let-7 계열의 hsa-let-7a-5p와 hsa-let-7c-5p)와 덜 명백한 예를 모두 찾을 수 있습니다. 예를 들어, 3행의 miRNA인 hsa-mir-107과 hsa-mir-103s-3p를 인터넷에서 검색하면, 이 두 miRNA가 골관절염, 낭포성 섬유증 및 기타 질병과 관련되어 있음을 논의하는 많은 논문을 찾을 수 있습니다.

표 C.4 목록 C.22의 유사도 질의 결과
<table><tr><td>소스</td><td>대상</td><td>유사도</td></tr><tr><td>hsa-let-7a-5p</td><td>hsa-let-7c-5p</td><td>1.0</td></tr><tr><td>hsa-let-7a-5p</td><td>hsa-let-7e-5p</td><td>1.0</td></tr><tr><td>hsa-mir-107</td><td>hsa-mir-103a-3p</td><td>1.0</td></tr><tr><td>hsa-mir-570-5p</td><td>hsa-mir-548ai</td><td>1.0</td></tr></table>

우리는 여기서 더 나아갈 수도 있습니다. 우리가 알고 있듯이, miRNA는 특정 mRNA에 간섭함으로써 유전자 발현을 조절하며, 이러한 조절이 비정상적일 때 병리가 발생할 수 있습니다. 두 개체가 공통으로 가지는 miRNA의 수를 바탕으로 표적 mRNA가 질병 (Disease)과 얼마나 유사하거나 친화적이라고 간주될 수 있는지 궁금해하는 것은 합리적입니다. 이러한 분석은 그 자체로도 흥미로울 뿐 아니라, 여러 데이터셋의 정보를 사용하여 서로 다른 엔터티(질병과 표적 (Target))를 비교할 수 있음을 보여 줍니다.

우리는 다시 GDS의 nodeSimilarity 함수를 사용할 것입니다. 그러나 이번에는 필터링된 버전입니다. 예를 들어 표적과 다른 표적 사이 또는 miRNA와 질병 사이가 아니라, 표적과 질병 사이의 유사도 관계에 관심이 있기 때문입니다. 다음 Cypher 질의는 유사도 분석의 기반이 될 인메모리 그래프 표현을 생성합니다.

```javascript
Listing C.23 Creating the in-memory graph
CALL gds.graph.project("Disease_Target_similarity",
["Target","MiRNA","Disease"],
{HAS_TARGET:{orientation:"UNDIRECTED"},
Makes sure we do not
RELATED_TO:{orientation:"UNDIRECTED"}, consider the direction
SIMILAR_TO:{orientation:"UNDIRECTED"}}) of the relationships
```

그래프가 메모리에 생성되면, 유사도 계산을 실행하고 결과를 평가할 수 있습니다.

목록 C.24 유사도 계산   
알고리즘이 질병   
CALL gds.nodeSimilarity.filtered.stream( 과 표적 노드 유형 간의 유사도만   
"Disease\_Target\_similarity", 고려하도록 지시합니다   
{sourceNodeFilter:"Disease",targetNodeFilter:"Target"}) <   
yield node1,node2, similarity   
WITH gds.util.asNode(node1) AS source, 질병과 표적이   
gds.util.asNode(node2) AS target, similarity 공통으로 가지는 miRNA 수를   
MATCH (source)-[]-(m:MiRNA)-[:HAS\_TARGET]-(target) ≤ 계산합니다   
WITH source, target, similarity, count(m) as miRNAs   
WHERE miRNAs > 10 <   
RETURN source.name AS source, target.name AS target, similarity, miRNAs   
ORDER BY similarity DESCENDING, source, target   
최소 10개의 miRNA를 공유하는   
질병과 표적만 유지합니다

유사도 질의의 결과는 표 C.5에 보고되어 있습니다. 목록 C.25의 질의는 첫 번째 행을 자세히 조사하며, 그 결과는 그림 C.20에 표시된 그래프와 같습니다.

표 C.5 listing C.24의 유사도 쿼리 결과
<table><tr><td>출처</td><td>대상</td><td>유사도</td><td>miRNA</td></tr><tr><td>수막종</td><td>NM_203347</td><td>0.047619048</td><td>11</td></tr><tr><td>수막종</td><td>NM_001031745</td><td>0.045454545</td><td>12</td></tr><tr><td>전립선 종양</td><td>NM_012316</td><td>0.030769231</td><td>16</td></tr><tr><td>전립선 종양</td><td>NM_001260491</td><td>0.030373832</td><td>13</td></tr></table>

![](images/61c91bf32da268ce7a95039aff15971ff5cc9d27bbeb6ed0d686d73af3078fd2.jpg)

이 예에서 대상 mRNA(NM_203347)와 연관된 miRNA의 거의 대부분이 수막종과 관련되어 있음을 볼 수 있습니다. 다시 말해, 이러한 유형의 발견은 의학적 관점에서 반드시 유의미할 필요는 없지만, 기계 학습 알고리즘이 사용할 수 있는 정보를 나타냅니다.

이 새로운 KG에서 실행할 마지막 분석은 4장에서 논의한 것처럼 Hetionet에 사용한 것과 동일한 분석으로, 질병에서 GO Process로 이어지는 관련 경로를 찾기 위해 차수 가중 경로 수 (degree-weighted path count, DWPC)를 사용합니다. 다시 상기하면, DWPC는 연결이 많은 노드에 페널티를 부여함으로써 많은 경로의 일부인 노드로 인해 분석이 편향되는 것을 피하도록 도와줍니다.

이 경우, 우리는 셀리악병에 익숙하고 결과를 효과적으로 평가할 수 있으므로 셀리악병을 참조 질병으로 사용합니다. 다음 목록은 셀리악병과 잠재적으로 관련된 관련 대상을 식별하기 위해 KG를 쿼리합니다.

Listing C.26 셀리악병에 연결된 관련 대상 검색   
MATCH path = (d:Disease)<-[:REGULATES|RELATED_TO]-(m)-[:HAS_TARGET]->(t)   
WHERE d.name = "celiac disease"   
WITH

[   
size([(d)<-[:REGULATES|RELATED_TO]-() | d]),   
size([()<-[:REGULATES|RELATED_TO]-(m) | m]),   
size([(m)-[:HAS_TARGET]->() | m]),   
size([()-[:HAS_TARGET]->(t) | t])   
]   
AS degrees, path, d, t   
WITH d.name as disease_name, t.name as target_name, count(path) as PC,   
sum(reduce(pdp = 1.0, d in degrees| pdp \* d ^ -0.4)) AS DWPC,   
size([(t)-[:HAS_TARGET]-() | t]) AS n_miRNA   
WHERE n_miRNA >= 5 and PC >= 2   
RETURN disease_name, target_name, PC, DWPC, n_miRNA   
ORDER BY DWPC desc   
LIMIT 10

결과는 표 C.6에 요약되어 있습니다. 이 결과는 셀리악병과 관련된 과학적 증거와 일치하기 때문에 흥미롭습니다. 예를 들어, 표의 첫 번째 대상은 NM_080601, 즉 “호모 사피엔스 protein tyrosine phosphatase non-receptor type 11 (PTPN11), transcript variant 2, mRNA”입니다. 연구에 따르면 단백질 티로신 포스파타아제가 면역계를 조절하는 역할과 이것이 만성 장 염증에 갖는 함의가 밝혀졌습니다 [34].

표의 타깃 2, 3, 5(NM_001224, NM_032982, NM_032983)는 모두 ”homo sapiens caspase 2, apoptosis-related cysteine peptidase,” CASP2의 변이체입니다. 글루텐 특이적 T 세포에 대한 단일세포 RNA-seq 조사 [35]를 포함한 여러 연구는 셀리악병의 근치적 치료 옵션으로서 글루텐 특이적 T 세포 제거를 위한 고유 타깃을 찾는 지식 기반을 제공합니다. 이러한 연구에서 연구자들은 테트라머 양성 세포에서 FAS, TRAIL, CASP2와 같은 여러 세포사멸 관련 유전자가 현저하게 상향 조절됨을 발견했으며, 이는 아마도 글루텐 항원에 의한 생체 내 활성화 때문일 수 있습니다. 이러한 발견은 글루텐 특이적 T 세포 제거를 위해 활성화 유도 세포사멸 (activation-induced cell death)을 사용하는 것을 장려합니다.

표 C.6 목록 C.26의 쿼리 결과
<table><tr><td>질병</td><td>타깃</td><td>PC</td><td>DWPC</td><td># miRNA</td></tr><tr><td>셀리악병</td><td>NM_080601</td><td>2</td><td>0.00417</td><td>25</td></tr><tr><td>셀리악병</td><td>NM_001224</td><td>2</td><td>0.00322</td><td>111</td></tr><tr><td>셀리악병</td><td>NM_032982</td><td>2</td><td>0.00318</td><td>114</td></tr><tr><td>셀리악병</td><td>NM_152617</td><td>2</td><td>0.00295</td><td>136</td></tr><tr><td>셀리악병</td><td>NM_032983</td><td>2</td><td>0.00278</td><td>160</td></tr><tr><td>셀리악병</td><td>NM_198926</td><td>3</td><td>0.00241</td><td>158</td></tr><tr><td>셀리악병</td><td>NM_019099</td><td>3</td><td>0.00234</td><td>169</td></tr><tr><td>셀리악병</td><td>NM_005235</td><td>2</td><td>0.00219</td><td>286</td></tr><tr><td>셀리악병</td><td>NM_052845</td><td>2</td><td>0.00210</td><td>138</td></tr><tr><td>셀리악병</td><td>NM_001142551</td><td>2</td><td>0.00209</td><td>138</td></tr></table>

표의 다른 타깃들이 발현하는 관련 유전자 변이체와 셀리악병을 연결하는 접근하기 쉬운 논문은 찾지 못했습니다. 그러나 이 질병을 둘러싼 활발한 연구를 고려하면, 상관관계가 드러날 가능성이 있습니다.

분석을 완료하기 위해 miRNA가 목록의 첫 번째 타깃과 어떻게 연결되는지 보여 주는 쿼리를 실행해 보겠습니다. 다음 목록은 셀리악병과 타깃 NM:080601을 연결하는 모든 경로를 검색하며, 결과는 그림 C.21에 나와 있습니다.

![](images/1bcac2621839e7cb431a21fc95b36707f2dc6da0d4c207e383d43980ea84f071.jpg)

우리가 입증했듯이, 여러 소스를 단일한 전체론적 지식 그래프 (KG)로 결합하면 다양한 관점에서 정보를 분석할 수 있습니다. 또한 DWPC와 같은 지표는 여러 맥락에 걸쳐 폭넓게 적용될 수 있습니다.

### 연습문제


해당 도메인에 관심이 있다면 대상 질병을 변경하여 앞서 수행한 질의를 다시 실행하고 결과를 평가하십시오.