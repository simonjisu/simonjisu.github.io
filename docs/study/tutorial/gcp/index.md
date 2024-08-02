---
title: "Learning Google Cloud Platform"
hide:
  - tags
tags:
  - "google cloud platform"
---

<figure markdown>
  ![HeadImg](https://lh3.googleusercontent.com/d/1fnSskBrNmd90GnsYoD2tjlcuw2eT2k7o){ class="skipglightbox" width="100%" }
  <figcaption>Reference: Google Cloud Platform</figcaption>
</figure>

Google Cloud Platform(GCP)[^1]는 아마존의 AWS[^2], 마이크로소프트의 Azure[^3]와 더불어 현재 삼 대(~~unofficial~~) 클라우드 플랫폼 서비스 중에 하나다. 

[^1]: [Google Cloud Platform](https://cloud.google.com/?hl=ko)
[^2]: [Amazon Web Cloud](https://aws.amazon.com/ko/?nc2=h_lg)
[^3]: [Microsoft Azure](https://azure.microsoft.com/ko-kr/)

<figure markdown>
  ![HeadImg](https://lh3.googleusercontent.com/d/1AACALNGjAVutIsEWiULu8iaQWTZTTEbS){ class="skipglightbox" width="100%" }
  <figcaption> 클라우스 서비스 3 대장 </figcaption>
</figure>

## 클라우드 플랫폼 서비스

> 클라우드 서비스란 타사 제공업체가 호스팅하여 인터넷을 통해 사용자에게 제공하는 인프라, 플랫폼 또는 소프트웨어를 말합니다[^4].

[^4]: [RedHat - What are Cloud Services](https://www.redhat.com/ko/topics/cloud-computing/what-are-cloud-services)

즉, 회사에서 물리적 하드웨어(컴퓨터, 하드 디스크 등등)를 이용자에게 인터넷을 통해 서비스하는 형태다. 구글은 전 세계에 여러 곳에 데이터 센터([구글의 데이터 센터의 위치보기](https://www.google.com/about/datacenters/locations/))를 두고 이를 서비스하고 있는데 이러한 데이터 센터의 위치를 **리전(region)**이라고 부른다. 당연히 통신 등 환경을 고려하여 자신의 위치와 가까운 곳에 리전을 선택하는 것이 좋지만 회사에서 유지보수 등의 이유로 비용의 차이도 발생한다(체감상 크게 차이는 나지 않는다고 한다).

클라우스 서비스를 사용하는 이유는 기업에서 직접 인프라를 관리할 필요 없이 사용에만 집중 할 수 있다. 하드웨어를 구입하고 설치하는 투자 비용을 정량적으로 계산할 수 있고, 개발환경을 구성하는 시간을 단축 시킬 수 있다. 기업의 입장에서 인프라 관리비용 또한 줄일 수 있어서 비용절감을 체험할 수 있다. 돈(:moneybag:)만 있다면 아주 큰 컴퓨팅 클러스터를 사용하여 파라미터가 많은 딥러닝 모델을 훈련시킬 수도 있다.

## 어떤 서비스를 제공?

인프라부터 시작해서 플랫폼, 완성형 소프트웨어등 다양한 서비스를 제공하고 있다. 구글 클라우드 플랫폼에서는 계산을 위한 Compute Engine, 빠르게 어플리케이션을 만들어주는 App Engine, 저장된 데이터를 검색, 분석하여 의사결정을 돕는 BigQuery 등 서비스를 제공하고 있다. 다양한 서비스와 타사의 주요 서비스를 비교한 문서는 [여기](https://cloud.google.com/free/docs/aws-azure-gcp-service-comparison?hl=ko)를 참고하길 바란다.

## 구성요소

### 서비스(Service)와 리소스(Resource)

클라우드 서비스에서는 소프트웨어 혹은 하드웨어 제품 등의 요소들이 **서비스(service)**라는 형태로 불리게 된다. 서비스를 통해 기반 리소스에 접근 할 수 있다. 다만 모든 리소스가 전역에서 공유되는 형태는 아니다. 같은 영역에서만 엑세스 가능한 일부 서비스들이 있다.

- **전역(Global) 리소스**: 사전에 구성된 디스크 이미지, 네트워크 등, 전역에서 엑세스 가능
- **리전(Region) 리소스**: 고정 외부 IP 주소 등, 같은 리전 내에서만 엑세스 가능
- **영역(Zone) 리소스**: VM 인스턴스, 디스크 등, 같은 영역 내에서만 엑세스 가능

![Image title](https://lh3.googleusercontent.com/d/1UTCMkWU-VXtwuLovioU9_GN44r8laLe-){ align=left width=100% }

### 프로젝트(Project)

프로젝트는 모든 구글 클라우드 서비스와 리소스를 담는 공간이다. 프로젝트 이름을 설정하면 고유의 번호, ID를 부여 받음며, 프로젝트가 사용하는 리소스에 따라 요금이 부과된다. 처음 사용하게 되면 $300 상당의 크레딧을 제공하고 있으며 서비스에 따라 무료 할당량도 존재한다.

조금 자세히 얘기하면 조직(Organization)에서 결제계정이 있고, 각 프로젝트 마다 결제계정을 설정하여 리소스를 관리 가능하다. 

## 참고하면 좋은 자료

- [사물궁이 잡학지식 - 클라우드란 무엇일까?](https://youtu.be/CpPEJyWwIgY)
- [서버리스 아키텍쳐란?](https://velopert.com/3543)
- [When to choose App Engine over Cloud Functions?](https://stackoverflow.com/questions/47057770/when-to-choose-app-engine-over-cloud-functions)
- [기술노트 - 서버 클라우드 / IT 인프라 / 서버리스](https://youtu.be/YSudWlx0o9I)