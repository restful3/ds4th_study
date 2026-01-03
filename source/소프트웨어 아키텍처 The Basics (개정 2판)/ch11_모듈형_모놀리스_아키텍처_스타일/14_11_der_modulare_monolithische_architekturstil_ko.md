---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 11장. 모듈러 모놀리식 아키텍처 스타일 (Kapitel 11. Der modulare monolithische Architekturstil)

이 작업은 AI를 사용하여 번역되었습니다. 피드백과 의견을 보내주시기 바랍니다: [translation-feedback@oreilly.com](mailto:translation-feedback@oreilly.com)

[Domain-Driven](https://oreil.ly/czIi5) Design (DDD)의 광범위한 도입과 도메인 분할에 대한 집중도 증가로 인해, *모듈러 모놀리식(modular monolithic)* 아키텍처 스타일은 2020년 이 책의 초판 출간 이후 매우 높은 인기를 얻게 되어, 우리는 2판에서 이를 설명하고 평가하는 장을 추가하기로 결정했습니다.

# 토폴로지 (Topologie)

이름에서 알 수 있듯이, 모듈러 모놀리식 아키텍처 스타일은 *모놀리식(monolithic)* 아키텍처입니다. 따라서 단일 소프트웨어 단위로 배포됩니다: WAR(Web Archive) 파일, .NET의 단일 어셈블리, Java 플랫폼의 EAR(Enterprise Archive) 파일 등입니다. 모듈러 모놀리스는 *도메인 분할을 갖춘* 아키텍처(기술적 기능이 아닌 비즈니스 영역에 따라 구성됨)로 간주되므로, 그 동형적 형태는 *단일 배포 단위*로 정의되며, *그 기능은 도메인 영역에 따라 그룹화됩니다*.

[그림](#page-1-0) 11-1은 모듈러 모놀리스의 전형적인 토폴로지를 보여줍니다.

모듈러 모놀리스의 특성을 이해하기 위해, 전통적인 계층형 아키텍처([10장](#page--1-0)에서 설명)를 생각해 보겠습니다. 그 컴포넌트는 *기술적* 기능에 따라 정의되고 구성됩니다: 프레젠테이션 계층, 비즈니스 계층, 영속성 계층 등. 예를 들어, 고객 프로필 관리를 위한 프레젠테이션 로직은 *com.app.presentation.customer.profile*이라는 네임스페이스의 컴포넌트로 표현될 수 있습니다. 네임스페이스의 세 번째 노드는 계층의 *기술적* 관심사를 나타냅니다(이 경우 프레젠테이션 계층).

<span id="page-1-0"></span>![](_page_1_Picture_2.jpeg)

그림 11-1. 모듈러 모놀리식 아키텍처에서는 기능이 도메인 영역별로 그룹화됩니다

반대로, 모듈러 모놀리식 컴포넌트는 주로 *도메인*에 따라 구성됩니다. 따라서 모듈러 모놀리식 아키텍처에서 고객 프로필 관리를 위한 동일한 컴포넌트는 *com.app.customer.profile* 네임스페이스로 표현됩니다. 이 경우 네임스페이스의 세 번째 노드는 기술적 측면보다는 *비즈니스* 측면을 나타냅니다. 컴포넌트의 복잡도에 따라, 네임스페이스는 *도메인 관심사* 이후 기술적 관심사에 따라 더 세분화될 수 있습니다. 예: *com.app.customer.profile.presentation* 또는 *com.app.customer.profile.business*.

## 스타일 특성 (Stil Besonderheiten)

이 아키텍처 스타일에서 도메인(또는 경우에 따라 서브도메인)은 *모듈(Module)*이라고 불립니다. 모듈은 두 가지 방식으로 구성될 수 있습니다. 가장 간단한 아키텍처는 *모놀리식 구조(monolithische Struktur)*로, 모든 모듈과 해당 논리적 컴포넌트가 동일한 코드베이스에 포함되며, 네임스페이스나 디렉토리 구조로 구분됩니다. 다소 복잡한 옵션은 *모듈러 구조(modulare Struktur)*로, 각 모듈이 독립적이고 자체 포함된 아티팩트(예: JAR 또는 DLL 파일)로 표현되며, 배포 시 모듈들이 모놀리식 소프트웨어 단위로 결합됩니다.

소프트웨어 아키텍처의 모든 것과 마찬가지로, 이 두 가지 구조적 옵션 중 선택은 많은 요인과 트레이드오프에 달려 있습니다. 다음 섹션에서는 두 가지 옵션을 소개하고 해당 트레이드오프를 논의합니다.

## <span id="page-3-1"></span>**모놀리식 구조 (Monolithische Struktur)**

모놀리식 구조에서는 시스템을 나타내는 모든 모듈이 단일 소스 코드 리포지토리에 포함됩니다. 개별 모듈의 모든 코드는 소프트웨어 출시 또는 릴리스 시 하나의 단위로 배포됩니다. 이 구조적 옵션은 [그림](#page-3-0) 11-2에 표시되어 있습니다. 각 모듈은 모듈을 구성하는 컴포넌트와 모든 서브도메인을 포함하는 자체 상위 디렉토리로 표현됩니다.

<span id="page-3-0"></span>![](_page_3_Figure_3.jpeg)

그림 11-2. 모놀리식 구조 옵션의 예

다음 네임스페이스는 [그림](#page-3-0) 11-2에 표시된 아키텍처에 대한 모듈러 모놀리스가 어떻게 보일 수 있는지 보여줍니다:

```
com.orderentry.orderplacement
com.orderentry.inventorymanagement
com.orderentry.paymentprocessing
com.orderentry.notification
com.orderentry.fulfillment
com.orderentry.shipping
```

이것은 모듈러 모놀리스의 가장 간단한 옵션입니다: 시스템의 모든 소스 코드가 한 곳에 있어 유지 관리, 테스트 및 배포가 더 쉽습니다. 그러나 아키텍처에서 사용되는 개별 모듈의 경계를 보호하기 위해 엄격한 거버넌스가 필요합니다(["거버넌스"](#page-12-0) 참조). 이 구조적 옵션은 간단하지만, 개발자들은 모듈 간에 코드를 과도하게 재사용하고 모듈 간 통신을 과도하게 허용하는 경향이 있습니다(["모듈 통신"](#page-6-0) 참조). 이러한 관행은 잘 설계된 모듈러 모놀리스를 구조화되지 않은 Big Ball of Mud로 전환할 수 있습니다.

### <span id="page-4-0"></span>**모듈러 구조 (Modularer Aufbau)**

모듈러 구조에서는 모듈이 독립 실행형 아티팩트(예: JAR 및 DLL 파일)로 표현되며, 배포 시 단일 배포 단위로 결합됩니다.

[그림](#page-5-0) 11-3은 Java 플랫폼의 JAR 파일을 사용한 이 옵션을 보여줍니다.

<span id="page-5-0"></span>![](_page_5_Picture_1.jpeg)

그림 11-3. JAR 파일을 사용한 모듈러 구조 옵션의 예

이 구조의 장점은 각 모듈이 자체 포함되어 있고 팀들이 별도의 모듈에서 작업할 수 있다는 것입니다(["팀 토폴로지 고려사항"](#page-16-0) 참조), 종종 이러한 모듈을 위한 팀 자체의 소스 코드 리포지토리 내에서도 가능합니다. 이 옵션은 모듈이 다른 모듈과 대체로 독립적일 때 잘 작동합니다. 또한 각 모듈이 다른 유형의 전문 지식이나 비즈니스 지식을 필요로 하는 대규모의 복잡한 시스템에 적합합니다. 모듈러 구조에서는 개발자들이 코드를 과도하게 재사용하거나 모듈 간 통신을 과도하게 허용할 가능성이 낮습니다(["모듈 통신"](#page-6-0) 참조). 이 옵션은 또한 모듈 간 경계를 더 명확하게 하고 개별 영역을 더 잘 분리합니다.

그러나 이 구조적 옵션은 서로 의존하는 모듈이 서로 통신해야 할 때 효과를 잃습니다. 이 경우 모놀리식 구조 접근 방식이 더 효과적입니다.

### <span id="page-6-0"></span>**모듈 통신 (Modul Kommunikation)**

이 아키텍처 스타일에서 모듈 간 통신은 결코 좋은 일이 아니지만, 많은 경우 필요하다는 것을 인정합니다. 예를 들어, [그림](#page-3-0) 11-2에 표시된 아키텍처에서 OrderPlacement 모듈은 InventoryManagement 모듈과 통신하여 주문된 항목의 재고를 조정하고 추가 처리를 수행해야 합니다(예: 재고가 너무 낮을 경우 더 많은 상품을 주문). 또한 주문에 대한 결제를 시작하기 위해 PaymentProcessing 모듈과 통신해야 합니다. 모듈 간 통신을 위한 두 가지 주요 옵션이 있으며, 다음 섹션에서 설명합니다.

### **피어 투 피어 접근 방식 (Peer-to-Peer-Ansatz)**

가장 간단한 솔루션은 모듈 간의 단순한 피어 투 피어 통신입니다. 이 접근 방식에서는 한 모듈의 클래스 파일이 다른 모듈의 클래스를 인스턴스화하고 작업을 수행하기 위해 해당 클래스의 필요한 메서드를 호출합니다([그림](#page-7-0) 11-4 참조).

<span id="page-7-0"></span>![](_page_7_Picture_1.jpeg)

그림 11-4. 모듈 간 피어 투 피어 통신

모놀리식 접근 방식의 문제점은 개발자가 다른 모듈의 모든 클래스를 인스턴스화하기가 *너무* 쉽다는 것입니다. 이로 인해 잘 구조화된 아키텍처에서 Big Ball of Mud 안티패턴으로 쉽게 전환될 수 있습니다([그림](#page--1-0) 9-1 참조).

그러나 모듈러 구조에서는 다른 모듈에 포함된 클래스가 소스 코드 리포지토리의 자체 디렉토리가 아닌 별도의 외부 아티팩트(JAR 또는 DLL 파일)에 있을 수 있습니다. 다른 모듈과 통신하는 모듈은 클래스 참조가 있을 때만 컴파일할 수 있습니다. 즉, 개발자는 *컴파일 타임에* 이러한 모듈 간의 종속성을 생성해야 합니다. 이 문제에 대한 일반적인 해결책은 모듈 간에 공통 인터페이스 클래스를 생성하는 것입니다(별도의 JAR 또는 DLL 파일에). 따라서 각 모듈은 다른 모듈과 독립적으로 컴파일할 수 있습니다. 어떤 경우든, 모듈러 구조를 사용할 때 모듈 간 통신이 너무 많으면 *DLL Hell* [안티패턴](https://oreil.ly/lx7U5)(또는 Java 플랫폼에서는 *JAR Hell* 안티패턴)으로 이어집니다.

### **중재자 접근 방식 (Mediator-Ansatz)**

*중재자(Mediator) 접근 방식*은 모듈 간의 추상화 계층으로 중재자 컴포넌트를 사용하여 모듈을 분리합니다. 중재자는 요청을 받아 적절한 모듈로 전달하는 오케스트레이터 역할을 합니다. [그림](#page-9-0) 11-5는 이 접근 방식을 보여줍니다.

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

그림 11-5. 중재자는 모듈을 분리하여 서로 통신할 필요가 없도록 합니다

주의 깊은 독자는 중재자 접근 방식이 모듈을 분리하지만, 각 모듈이 효과적으로 중재자에 결합되어 있다는 것을 알 수 있을 것입니다. 이 접근 방식은 *모든* 결합과 종속성을 제거하지는 않지만, 아키텍처를 단순화하고 모듈을 서로 독립적으로 유지합니다. 다른 모듈의 기능을 호출하기 위해 일종의 API 또는 인터페이스가 필요한 것은 종속 모듈이 아니라 *중재자*임을 유의하십시오.

# 데이터 토폴로지 (Daten-Topologien)

모듈러 모놀리식 아키텍처는 일반적으로 단일 소프트웨어 단위로 배포되므로, 일반적으로 모놀리식 데이터베이스 토폴로지를 기반으로 합니다. 단일 데이터베이스를 사용하면 데이터가 공유되므로 모듈 간 통신을 줄이는 데 도움이 됩니다. 그러나 모듈이 서로 독립적이고 특정 기능을 수행하는 경우, 아키텍처 자체가 모놀리식이더라도 특정 컨텍스트 데이터가 있는 자체 데이터베이스를 가질 수 있습니다. [그림](#page-10-0) 11-6은 이 두 가지 데이터베이스 토폴로지 옵션을 보여줍니다.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

그림 11-6. 데이터는 모놀리식일 수 있거나 모듈이 자체 데이터베이스를 가질 수 있습니다

# 클라우드 고려사항 (Überlegungen zur Cloud)

모듈러 모놀리식 아키텍처는 클라우드 환경에 배포할 수 있지만(특히 작은 시스템의 경우), 일반적으로 클라우드 배포에 적합하지 않습니다. 모놀리식 특성으로 인해 클라우드 환경이 제공하는 온디맨드 프로비저닝의 이점을 덜 활용할 수 있습니다. 그럼에도 불구하고, 이 아키텍처 스타일로 구현된 작은 시스템도 파일 저장, 데이터베이스 및 메시징과 같은 많은 클라우드 서비스를 활용할 수 있습니다.

## 일반적인 위험 (Gemeinsame Risiken)

모든 모놀리식 시스템과 마찬가지로, 모듈러 모놀리식 아키텍처의 주요 위험은 적절하게 유지 관리, 테스트 및 구현하기에 너무 커질 수 있다는 것입니다. 모놀리식 아키텍처 자체가 나쁜 것은 아니지만, 너무 커지면 문제가 발생합니다. "너무 크다"는 것이 무엇을 의미하는지는 시스템마다 다르지만, 시스템이 너무 클 수 있다는 경고 신호는 다음과 같습니다:

- 변경을 수행하는 데 너무 오래 걸립니다.
- 시스템의 한 영역이 변경되면 다른 영역이 예기치 않게 중단됩니다.
- 팀 멤버들이 변경 사항을 구현할 때 서로 방해합니다.
- 시스템이 시작되는 데 너무 오래 걸립니다.

또 다른 위험은 코드의 과도한 재사용입니다. 코드의 재사용과 공유는 소프트웨어 개발의 필수 부분이지만, 이 아키텍처 스타일에서 코드를 과도하게 재사용하면 모듈 경계가 흐려지고 아키텍처가 *구조화되지 않은 모놀리스(unstrukturierter Monolith)*의 위험한 영역으로 이동합니다: 코드가 너무 강하게 상호 의존적이어서 더 이상 풀 수 없는 모놀리식 아키텍처.

모듈 간의 과도한 통신은 이 아키텍처 스타일의 또 다른 위험입니다. 이상적으로 모듈은 독립적이고 자체 포함되어야 합니다. 이미 언급했듯이, 일부 모듈이 다른 모듈과 통신하는 것은 정상적이며(때로는 필요하기도 합니다), 특히 복잡한 워크플로 내에서 그렇습니다. 그러나 모듈 간 통신이 너무 많으면 도메인이 올바르게 정의되지 않았다는 좋은 표시입니다. 이러한 경우 복잡한 워크플로와 상호 의존성을 고려하여 도메인을 재정의하는 것이 좋습니다.

## <span id="page-12-0"></span>거버넌스 (Governance)

모듈러 모놀리식 스타일의 주요 아티팩트는 특정 도메인 또는 서브도메인을 나타내는 *모듈*이며, 일반적으로 디렉토리 구조 또는 네임스페이스(또는 Java 플랫폼의 패키지 구조)로 표현됩니다. 따라서 아키텍트가 적용할 수 있는 첫 번째 자동화된 거버넌스 형태 중 하나는 아키텍처에서 사용되는 모듈의 정의와 준수 보장입니다.

자동화된 거버넌스 검사를 작성하기 위해, 아키텍트는 Java 플랫폼용 [ArchUnit](https://archunit.org/), .NET 플랫폼용 [ArchUnitNet](https://oreil.ly/-hjOC) 및 [NetArchTest](https://oreil.ly/4e-2j), Python용 [PyTestArch](https://oreil.ly/lWKVt), TypeScript 및 JavaScript용 [TSArch](https://oreil.ly/Fk4OG)를 포함한 다양한 도구를 사용합니다. [예제](#page-13-0) 11-1의 의사 코드는 [그림](#page-3-0) 11-2에 표시된 아키텍처 예제의 모든 소스 코드가 시스템에서 정의된 각 모듈을 나타내는 나열된 네임스페이스 중 하나에 속하는지 확인합니다.

### <span id="page-13-0"></span>**예제 11-1. 코드가 시스템의 정의된 모듈을 따르는지 확인하는 의사 코드**

```
# The following namespaces represent the modules in the system
LIST module_list = {
   com.orderentry.orderplacement,
   com.orderentry.inventorymanagement,
   com.orderentry.paymentprocessing,
   com.orderentry.notification,
   com.orderentry.fulfillment,
   com.orderentry.shipping
   }
# Get the list of namespaces in the system
LIST namespace_list = get_all_namespaces(root_directory)
# Make sure all the namespaces start with one of the listed modules
FOREACH namespace IN namespace_list {
   IF NOT namespace.starts_with(module_list) {
      send_alert(namespace)
   }
}
```

개발자가 정의된 모듈 및 해당 네임스페이스(또는 디렉토리) 외부에 추가 고수준 네임스페이스 또는 디렉토리를 생성하면, 소스 코드가 아키텍처와 일치하지 않는다는 것을 나타내는 경고를 받습니다.

이 거버넌스 형태는 이 아키텍처 스타일의 모놀리식 구조 옵션(["모놀리식 구조"](#page-3-1) 참조)에서 잘 작동하지만, 모듈러 구조 옵션(["모듈러 구조"](#page-4-0) 참조)에서는 코드가 동일한 모놀리식 소스 코드 리포지토리에 포함되지 않을 수 있으므로 어려움이 있습니다. 모듈러 구조에서는 [예제](#page-14-0) 11-2에 표시된 것처럼 각 모듈을 개별적으로 테스트해야 합니다.

### <span id="page-14-0"></span>**예제 11-2. InventoryManagement 모듈 검증을 위한 의사 코드**

```
# Get the list of namespaces in the system
LIST namespace_list = get_all_namespaces(root_directory)
# Make sure all the namespaces start with com.orderentry.inventorymanagement
FOREACH namespace IN namespace_list {
   IF NOT namepace.starts_with("com.orderentry.inventorymanagement") {
      send_alert(namespace)
   }
}
```

모듈러 모놀리식 아키텍처를 관리하는 또 다른 방법은 모듈 간 통신의 범위를 제어하는 것입니다. "너무 많은" 통신이 무엇인지 정의하는 것은 매우 주관적이며 시스템마다 다르지만, 대부분의 경우 아키텍트는 모듈 간 종속성 수를 최소화하려고 노력해야 합니다. [예제](#page-15-0) 11-3은 최대 총 종속성이 5개 이하의 통신 지점(또는 결합 지점)이 되도록 보장하는 의사 코드를 보여줍니다.

### <span id="page-15-0"></span>**예제 11-3. 특정 모듈의 총 종속성 수를 제한하기 위한 의사 코드**

```
# Walk the directory structure, gathering modules and the source code files
# contained within those modules
LIST module_list = {
   com.orderentry.orderplacement,
   com.orderentry.inventorymanagement,
   com.orderentry.paymentprocessing,
   com.orderentry.notification,
   com.orderentry.fulfillment,
   com.orderentry.shipping
   }
MAP module_source_file_map
FOREACH module IN module_list {
  LIST source_file_list = get_source_files(module)
  ADD module, source_file_list TO module_source_file_map
}
# Determine how many references exist for each source file and send an alert if
# the system's total dependency count is greater than 5
FOREACH module, source_file_list IN module_source_file_map {
  FOREACH source_file IN source_file_list {
    incoming count = used_by_other_module(source_file, module_source_file_map) {
    outgoing_count = uses_other_module(source_file) {
    total_count = incoming count + outgoing count
  }
  IF total_count > 5 {
    send_alert(module, total_count)
  }
}
```

자동 관리의 마지막 형태는 특정 모듈이 다른 모듈과 통신할 수 없도록 하여 모듈이 서로 독립적으로 유지되도록 하는 것입니다. 예를 들어, [그림](#page-3-0) 11-2에서 OrderPlacement 모듈은 Shipping 모듈과 통신해서는 안 됩니다.

[예제](#page-16-1) 11-4는 이 종속성을 관리하기 위한 Java의 ArchUnit 코드를 보여줍니다.

### <span id="page-16-1"></span>**예제 11-4. 특정 모듈 간 종속성 제한을 관리하기 위한 ArchUnit 코드**

```
public void order_placement_cannot_access_shipping() {
   noClasses().that()
   .resideInAPackage("..com.orderentry.orderplacement..")
   .should().accessClassesThat()
   .resideInAPackage("..com.orderentry.shipping..")
   .check(myClasses);
}
```

# <span id="page-16-0"></span>팀 토폴로지 고려사항 (Überlegungen zur Team-Topologie)

모듈러 모놀리스는 도메인별 아키텍처로 간주되므로, 팀도 도메인에 따라 정렬될 때 가장 잘 작동합니다(예: 전문화된 교차 기능 팀). 도메인 관련 요구 사항이 발생하면, 도메인 지향적이고 교차 기능적인 팀이 프레젠테이션 로직에서 데이터베이스까지 해당 기능을 함께 작업할 수 있습니다. 반대로, 기술 범주에 따라 구성된 팀(예: UI 팀, 백엔드 팀, 데이터베이스 팀 등)은 이 아키텍처 스타일과 잘 작동하지 않습니다. 주로 영역별 분할 때문입니다. 기술적으로 구성된 팀에 도메인 요구 사항을 할당하려면 많은 커뮤니케이션과 협업이 필요하며, 이는 종종 어려운 것으로 입증됩니다.

다음은 ["팀 토폴로지 및 아키텍처"](#page--1-1)에서 설명한 특정 팀 토폴로지를 모듈러 모놀리스 스타일과 정렬하기 위한 몇 가지 고려 사항입니다:

#### *스트림 정렬 팀 (Auf den Strom ausgerichtete Teams)*

스트림 정렬 팀은 일반적으로 처음부터 끝까지 시스템을 통한 흐름을 제어하며, 이는 모듈러 모놀리스의 모놀리식하고 일반적으로 자체 포함된 형태와 잘 맞습니다.

#### *지원 팀 (Teams befähigen)*

높은 수준의 모듈성과 관심사 분리로 인해 팀 토폴로지도 잘 작동합니다. 전문가 및 영역 간 팀 멤버는 다른 기존 모듈에 영향을 주지 않고 시스템에 추가 모듈을 도입하여 제안하고 실험을 수행할 수 있습니다.

#### *복잡한 하위 시스템 팀 (Teams mit komplizierten Subsystemen)*

모듈러 모놀리식 아키텍처의 각 모듈은 일반적으로 도메인 또는 서브도메인과 관련된 특정 작업을 수행합니다(예: PaymentProcessing). 이것은 복잡한 하위 시스템 팀 토폴로지와 잘 작동합니다. 다양한 팀 멤버가 다른 팀 멤버(및 모듈)와 독립적으로 복잡한 도메인 또는 서브도메인 처리에 집중할 수 있기 때문입니다.

#### *플랫폼 팀 (Plattform-Teams)*

개발자는 이 아키텍처 스타일이 제공하는 높은 수준의 모듈성으로 인해, 공통 도구, 서비스, API 및 작업을 사용하여 플랫폼 팀 토폴로지의 이점을 활용할 수 있습니다.

# 스타일 특성 (Stilmerkmale)

[그림](#page-20-0) 11-7의 표에서 별 1개 등급은 특정 아키텍처 특성이 아키텍처에서 잘 지원되지 않음을 의미하며, 별 5개 등급은 아키텍처 특성이 아키텍처 스타일의 가장 강력한 특성 중 하나임을 의미합니다. 스코어카드에 포함된 특성은 [4장](#page--1-0)에서 설명되고 정의됩니다.

모듈러 모놀리식 아키텍처 스타일은 애플리케이션 로직이 모듈로 분할되므로 *도메인 분할(domain-partitioned)* 아키텍처입니다. 일반적으로 모놀리식 배포로 구현되므로, 아키텍처 퀀텀은 일반적으로 1입니다.

전체 비용, 단순성 및 모듈성은 모듈러 모놀리식 아키텍처 스타일의 주요 강점입니다. 이러한 아키텍처는 모놀리식이므로 분산 아키텍처와 관련된 복잡성을 나타내지 않습니다. 더 간단하고 이해하기 쉬우며 구축 및 유지 관리 비용이 상대적으로 저렴합니다. 아키텍처의 모듈성은 도메인과 서브도메인을 나타내는 다양한 모듈 간의 관심사 분리를 통해 달성됩니다.

배포 가능성과 테스트 가능성은 별 2개로 평가되지만, 모듈러 모놀리식 아키텍처는 모듈성으로 인해 계층형 아키텍처보다 약간 높습니다. 그럼에도 불구하고 이 아키텍처도 모놀리스입니다: 의식, 위험, 배포 빈도 및 테스트 완전성이 이러한 값에 부정적인 영향을 미칩니다.

모듈러 모놀리식 아키텍처의 탄력성과 확장성은 매우 낮으며(별 1개), 이는 주로 모놀리식 구현 때문입니다. 모놀리스 내의 특정 기능을 다른 기능보다 더 확장하는 것이 가능하지만, 이 노력은 일반적으로 이 아키텍처가 잘 맞지 않는 매우 복잡한 설계 기술(멀티스레딩, 내부 메시징 및 기타 병렬 처리 관행 등)을 필요로 합니다.

|             | Architectural characteristic | Star rating                      |
|-------------|------------------------------|----------------------------------|
|             | Overall cost                 | \$                               |
| Structural  | Partitioning type            | Domain                           |
|             | Number of quanta             | 1                                |
|             | Simplicity                   | ***                              |
|             | Modularity                   | $\stackrel{\wedge}{\Rightarrow}$ |
| Engineering | Maintainability              | $\stackrel{\wedge}{\Rightarrow}$ |
|             | Testability                  | $\stackrel{\wedge}{\Rightarrow}$ |
|             | Deployability                | $\stackrel{\wedge}{\Rightarrow}$ |
|             | Evolvability                 | $\stackrel{\wedge}{\Rightarrow}$ |
| Operational | Responsiveness               | <b>☆☆☆</b>                       |
|             | Scalability                  | $\overset{\bigstar}{}$           |
|             | Elasticity                   | $\stackrel{\bigstar}{\sim}$      |
|             | Fault tolerance              | $\stackrel{\bigstar}{\sim}$      |

모듈러 모놀리식 아키텍처는 모놀리식 구현이므로 내결함성을 지원하지 않습니다. 아키텍처의 작은 부분이 메모리 부족 상태를 일으키면 전체 애플리케이션 단위가 충돌합니다. 또한 대부분의 모놀리식 애플리케이션과 마찬가지로, 높은 평균 복구 시간(MTTR)으로 인해 전체 가용성이 영향을 받으며, 시작 시간은 일반적으로 분 단위로 측정됩니다.

### **언제 사용할 것인가 (Wann verwendet man)**

단순성과 낮은 비용으로 인해, 모듈러 모놀리식 아키텍처는 예산과 시간이 제한적일 때 좋은 선택입니다. 또한 새로운 시스템을 시작할 때도 좋은 선택입니다. 시스템의 아키텍처 방향이 아직 불분명할 때, 모듈러 모놀리스로 시작하여 나중에 서비스 기반([14장](#page--1-0) 참조) 또는 마이크로서비스([18장](#page--1-0) 참조)와 같은 더 복잡하고 비싼 분산 아키텍처 스타일로 전환하는 것이 직접 분산 아키텍처로 뛰어드는 것보다 종종 더 효과적입니다.

모듈러 모놀리스는 도메인 지향 팀, 예를 들어 전문화된 교차 기능 팀에게도 좋은 선택입니다. 각 팀은 처음부터 끝까지 아키텍처 내의 특정 모듈에 집중할 수 있으며 다른 팀과 최소한의 조정만 필요합니다. 이 아키텍처 스타일은 시스템에 대한 대부분의 변경이 도메인 기반인 상황에도 적합합니다(예: 고객의 위시리스트 항목에 만료 날짜 추가).

모듈러 모놀리스는 도메인 분할 아키텍처이므로, [DDD](https://oreil.ly/czIi5)를 다루는 팀에 적합합니다.

### **언제 사용하지 말아야 할 것인가 (Wann man nicht verwenden sollte)**

이 아키텍처 스타일을 사용하지 않는 주요 이유는 시스템 또는 제품이 확장성, 탄력성, 가용성, 내결함성, 응답성 및 성능과 같은 특정 운영 특성에 대해 높은 수준을 요구할 때입니다. 대부분의 모놀리식 아키텍처와 마찬가지로, 모듈러 모놀리스는 이러한 요구 사항에 적합하지 않습니다.

대부분의 변경이 기술 지향적일 때, 예를 들어 사용자 인터페이스나 데이터베이스 기술이 지속적으로 교체될 때는 모듈러 모놀리스 사용을 피하십시오. 이 아키텍처는 다양한 영역으로 분할되어 있으므로, 이러한 변경은 각 모듈에 영향을 미치며 일반적으로 개별 영역의 팀 간에 높은 수준의 커뮤니케이션과 조정 노력이 필요합니다. 이러한 상황에서는 계층형 아키텍처([10장](#page--1-0) 참조)가 훨씬 더 나은 선택입니다.

# 예제 및 사용 사례 (Beispiele und Anwendungsfälle)

EasyMeals는 힘든 하루를 보낸 후 항상 요리할 시간이 없는 전문직 종사자를 대상으로 하는 새로운 배달 레스토랑입니다. 배고픈 고객은 온라인으로 맛있는 식사를 주문할 수 있으며 한 시간 이내에 집으로 배달받습니다.

작고 지역적인 레스토랑으로서 확장성이나 응답성에 대한 높은 요구 사항이 없습니다. 그리고 예산이 제한적이므로, 정교한 소프트웨어 시스템에 많은 돈을 쓰고 싶지 않습니다. 이 비즈니스 문제의 형태는 모듈러 모놀리스를 EasyMeals에 좋은 선택으로 만듭니다.

[그림](#page-24-0) 11-8은 모듈러 모놀리식 아키텍처 스타일을 사용할 때 EasyMeals의 간단한 레스토랑 관리 시스템이 어떻게 보일 수 있는지 보여줍니다.

![](_page_24_Figure_0.jpeg)

고객은 자체 사용자 인터페이스를 통해 PlaceOrder 및 PaymentProcessing 모듈에 액세스합니다. 다음 네임스페이스는 이 시스템의 개별 모듈을 나타냅니다:

```
com.easymeals.placeorder
com.easymeals.payment
com.easymeals.prepareorder
com.easymeals.delivery
com.easymeals.recipes
com.easymeals.inventory
```

PlaceOrder 모듈을 사용하면 각 고객이 메뉴를 보고, 항목을 선택하고, 이름, 주소 및 결제 정보를 추가하고, 주문을 제출할 수 있습니다. 이 모듈의 컴포넌트는 다음 네임스페이스로 표현할 수 있으며, 소스 코드는 이러한 주요 기능 각각을 구현합니다:

```
com.easymeals.placeorder.menu
com.easymeals.placeorder.shoppingcart
com.easymeals.placeorder.customerdata
com.easymeals.placeorder.paymentdata
com.easymeals.placeorder.checkout
```

이 예는 모듈러 모놀리스의 모듈이 하나에서 여러 개의 *컴포넌트*로 구성될 수 있음을 보여줍니다([8장](#page--1-0) 참조).

PaymentProcessing 모듈은 결제를 담당합니다. EasyMeals는 신용 카드, 직불 카드 및 PayPal을 허용합니다. 이 아키텍처의 모듈성은 추가 결제 유형(예: 로열티 포인트)을 쉽게 추가할 수 있게 합니다. 고객은 이 정보를 PlaceOrder 모듈에 입력하고, 이 모듈이 PaymentProcessing 모듈로 전달합니다. 이 모듈의 컴포넌트는 다음 네임스페이스로 표현될 수 있습니다:

```
com.easymeals.payment.creditcard
com.easymeals.payment.debitcard
com.easymeals.payment.paypal
```

주문이 결제되면, PlaceOrder 모듈은 주방 직원에게 전체 주문을 표시하는 PrepareOrder 모듈과 통신합니다. 조리 후, 주방 직원은 주문을 배달 준비 완료로 표시하고 Delivery 모듈로 보냅니다. 다음 네임스페이스는 PrepareOrder 모듈 내의 컴포넌트를 나타냅니다:

```
com.easymeals.prepareorder.displayorder
com.easymeals.prepareorder.ready
```

Delivery 모듈은 주문에 배달원을 할당하고 배달 주소를 제공합니다. 배달원이 주문을 배달 완료로 표시하여 해당 주문의 수명 주기를 종료할 수 있습니다. 또한 모든 문제를 기록할 수 있습니다(예: 입구 게이트에 공격적인 개가 있거나 고객이 집에 없는 경우). 다음 네임스페이스는 Delivery 모듈의 컴포넌트를 나타냅니다:

```
com.easymeals.delivery.assign
com.easymeals.delivery.issues
com.easymeals.delivery.complete
```

Recipes 모듈을 사용하면 요리사와 관리 직원이 메뉴에 항목을 추가하고 각 메뉴 항목에 대한 재료 및 측정값 목록을 유지 관리할 수 있습니다. 다음 네임스페이스는 Recipes 모듈의 컴포넌트를 나타냅니다:

```
com.easymeals.recipes.view
com.easymeals.recipes.maintenance
```

마지막으로, IngredientsInventory 모듈은 메뉴의 레시피에 충분한 재료가 있는지 확인합니다. 이 모듈은 다른 모듈보다 다소 복잡합니다: 판매를 예측하여 일주일 동안의 재료 조달을 자동화하는 정교한 AI 컴포넌트가 있습니다.

다음 네임스페이스는 IngredientsInventory 모듈 내의 컴포넌트를 나타냅니다:

```
com.easymeals.inventory.maintenance
com.easymeals.inventory.forecasting
com.easymeals.inventory.ordering
com.easymeals.inventory.suppliers
com.easymeals.inventory.invoices
```

그게 전부입니다! 모듈러 모놀리스의 단순성과 모듈성 정도는 버그를 수정하거나 새로운 기능을 추가하기 위해 코드를 찾고 유지 관리하는 것을 비교적 쉽게 만듭니다. 이것은 이 간단하고 직관적인 아키텍처 스타일의 강점을 보여줍니다.