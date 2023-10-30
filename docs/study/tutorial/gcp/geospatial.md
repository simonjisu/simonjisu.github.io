---
title: "Big Query Geospatial Data"
hide:
  - tags
tags:
  - "bigquery"
  - "google cloud platform"
---

구글 빅쿼리에서 지리정보 데이터를 다루는 방법에 대해 알아본다. [GEOGRAPHY](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types?hl=ko#geography_type)라는 타입의 데이터를 사용해서 좌표계의 점, 선, 면등을 다룬다. 

지구상의 한 점은 **위도(latitude)**와 **경도(longtitude)**로 표현할 수 있다. 위도는 북쪽으로부터의 각도이고, 경도는 동쪽으로부터의 각도이다. 위도와 경도는 각각 -90도에서 90도 사이의 값을 가진다. 경도는 -180도에서 180도 사이의 값을 가진다.

예를 들어 미국의 자유 여신상의 좌표는 `(40.69046171976406, -74.04444975893499)`으로 표현된다. ~~우리나라는 지도 데이터 반출 이슈 때문인지 좌표 조회가 안된다.~~

<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d17616.10519528353!2d-74.04616637254513!3d40.68577575847555!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89c25090129c363d%3A0x40c6a5770d25022b!2z7Iqk7YWM7LiEIOyYpOu4jCDrpqzrsoTti7Ag6rWt6rCA6riw64WQ66y8!5e0!3m2!1sko!2skr!4v1680954842727!5m2!1sko!2skr" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>

GeoJSON[^1]은 지리 정보데이터를 저장하는 방법 중에 하나며, JSON 형식으로 표현된다. GeoJSON은 지리 정보데이터를 표현하는데 필요한 기본적인 요소들을 정의하고 있다. 여기 뿐만 아니라 다른 곳에서도 많이 쓰이기 때문에 알아 두는 것이 좋다.

```json
{ "type": "Point", "coordinates": [-121,41] }
```

[^1]: [https://geojson.org/](https://geojson.org/)

## 예제1: 허리케인 경로 표시

Big Query에서 제공하는 공개 데이터세트 Global Hurricane Tracks (IBTrACS)로 허리케인의 경로를 표시해보자. 이 데이터세트는 1851년부터 2019년까지의 허리케인의 경로를 저장하고 있다. 튜토리얼은 [여기](https://cloud.google.com/bigquery/docs/visualize-geospatial-data?hl=ko)를 참고했다.

Big Query 탐색기에서 `+추가`를 누르고 공개 데이터 세트에서 Global Hurricane Tracks를 검색하여 데이터를 추가한다.

![Image title](https://drive.google.com/uc?id=19N7VANFTk-YNqfNGi16NJhcOEGqgNowb){ align=left width=40% }

공개 데이터 세트 `noaa_hurricanes`의 `hurricanes` 테이블을 선택하고 `스키마`를 눌러 데이터의 타입을 확인한다. 그리고 `쿼리`를 눌러서 아래의 쿼리를 실행한다. 

```sql
-- # SELECT 절은 폭풍의 모든 날씨 데이터를 선택하고 ST_GeogPoint 함수를 사용하여 latitude 및 longitude 열의 값을 GEOGRAPHY 유형(점)으로 변환합니다.
SELECT
    ST_GeogPoint(longitude, latitude) AS point,  -- 위도, 경도
    name, -- 허리케인 이름
    iso_time,  -- 시간
    dist2land,  --육지와의 거리
    usa_wind,  -- 최대 지속 풍속(노트): 0 - 300 kts.
    usa_pressure,  -- 최소 중심 기압(밀리바)
    usa_sshs,  -- Saffir-Simpson Hurricane Wind Scale: -5 ~ 5
    (usa_r34_ne + usa_r34_nw + usa_r34_se + usa_r34_sw)/4 AS radius_34kt, -- 34kt 속도 평균 반경
    (usa_r50_ne + usa_r50_nw + usa_r50_se + usa_r50_sw)/4 AS radius_50kt -- 50kt 속도 평균 반경
FROM
    `bigquery-public-data.noaa_hurricanes.hurricanes`
-- WHERE 절은 2017년 허리케인 시즌의 허리케인 마리아를 따라 이 데이터를 대서양의 점으로 필터링합니다.
WHERE
    name LIKE '%MARIA%'  -- 마리아
    AND season = '2017'  -- 2017년
    AND ST_DWithin(
        -- 폴리곤 안의 모든 점들: https://wktmap.com/?484becc9
        ST_GeogFromText('POLYGON((-179 26, -179 48, -10 48, -10 26, -100 -10.1, -179 26))'), 
        ST_GeogPoint(longitude, latitude),  -- 좌표들2
        10  -- 거리
    )
-- ORDER BY 절은 점을 순서대로 나열하여 폭풍의 경로를 시간순으로 형성합니다.
ORDER BY
    iso_time ASC
```

* `ST_GeogPoint`[^2]: 좌표, (경도 longtitude, 위도 latitude)로 지정
* `ST_DWithin`[^3]: 두 Geometry가 지정된 거리 내에 있는지 판별, 거리 내에 있으면 TRUE, 아니면 FALSE 반환
* `ST_GeogFromText`[^4]: WKT(Well-Known Text)[^6] 형식의 문자열을 지정된 GEOGRAPHY 유형으로 변환

[^2]: [ST_GeogPoint](https://cloud.google.com/bigquery/docs/reference/standard-sql/geography_functions#st_geogpoint)
[^3]: [ST_DWithin](https://cloud.google.com/bigquery/docs/reference/standard-sql/geography_functions#st_dwithin)
[^4]: [ST_GeogFromText](https://cloud.google.com/bigquery/docs/reference/standard-sql/geography_functions#st_geogfromtext)
[^6]: [Well-known text representation of geometry](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry)

??? note "Quadraphonic Wind[^2]"

    ![Image title](https://www.noaa.gov/sites/default/files/2022-11//ll_quadwind_soln5.png){ align=left width=50% }

    노트(kn 혹은 kt): 1시간에 1 해리(nm 혹은 nmile)를 진행하는 속도

    * 파랑색 원: 태풍의 눈
    * 초록색: 64노트(118km/h)의 방향별 풍속 최대 영향 반경
    * 노랑색: 50노트(93km/h)의 방향별 풍속 최대 영향 반경
    * 보라색: 34노트(63km/h)의 방향별 풍속 최대 영향 반경

[^5]: [Learning Lesson: Quadraphonic Wind](https://www.noaa.gov/jetstream/tc-tcm/learning-lesson-quadraphonic-wind)

실행 후 해당 데이터를 [Geo Viz](https://bigquerygeoviz.appspot.com/?hl=ko)에서 쿼리 결과를 시각화한다. 쿼리를 실행하고 시각화 형식을 지정할 수 있다. 옵션은 다음과 같다.

| Option     | Description  |
| ----------- | ------------ |
| **fillColor**   | 다각형 또는 점의 채우기 색상. 예를 들어 'linear' 또는 'interval' 함수는 숫자 값을 색상 그라디언트에 매핑하는 데 사용할 수 있음. |
| **fillOpacity** | 다각형 또는 점의 채우기 불투명도. 값은 0에서 1 사이. 여기서 0은 투명이고, 1은 불투명. |
| **strokeColor** | 다각형 또는 선의 획이나 윤곽선 색상. |
| **strokeOpacity** | 다각형 또는 선의 획이나 윤곽선 불투명도. 값은 0에서 1 사이. 여기서 0은 투명이고, 1은 불투명. |
| **strokeWeight** | 다각형 또는 선의 획 또는 윤곽선 너비(픽셀 단위). |
| **circleRadius** | 점을 나타내는 원의 반지름(미터 단위). 예를 들어 산점도 스타일을 만들기 위해 'linear' 함수를 사용하여 숫자 값을 점 크기에 매핑할 수 있음. |

!!! info "설정"

    === "fillColor"
            
        ![Image title](https://drive.google.com/uc?id=19R4YTCWSw7GbQA5z7wqJ88r5ICuPSOqS){ class="skipglightbox" width=80% }

    === "fillOpacity"

        ![Image title](https://drive.google.com/uc?id=19UI6-JHAGc0YqU9OkVz_1bNcxDg0UsvL){ class="skipglightbox" width=100% }

    === "circleRadius"

        ![Image title](https://drive.google.com/uc?id=19RES057I0jjPzi5GY2BnlVhIo1_xLAZp){ class="skipglightbox" width=80% }

## 예제2: 스타벅스와 생활인구 데이터 시각화

스타벅스 매장의 위치 데이터와 생활인구를 시각화해보자. 생활인구란 통신데이터로 특정 시점에 개인이 위치한 지역을 집계한 '현주인구'를 말한다. 시간대에 따라 변화하는 인구의 규모로 지역간 특성을 추측해 볼 수 있는 유용한 데이터다. 서울 열린데이터 광장[^2]에서 데이터를 받을 수 있다.

[^7]: [서울 열린데이터 광장 - 서울 생활인구](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do)

2022년 데이터를 받아서 12월 25일 오후 3시대 각 20대 남녀 생활인구를 스타벅스 매장의 좌표 위치에 시각화해본다. 전처리 된 데이터는 [여기](https://github.com/simonjisu/bkms2_2023spring/tree/main/03_geospatial)에서 받을 수 있다.

`starbucks`라는 데이터 세트를 생성해주고 두 개의 데이터를 업로드한다. 각각 테이블의 이름은 `starbucks_kr`와 `people_1225`으로 한다. 그리고 시각화 데이터를 준비하기 위해 다음의 쿼리를 실행한다.

```sql
CREATE VIEW `[project_id]`.`starbucks`.`new_view` AS (
    SELECT 
    ST_GEOGPOINT(s.long, s.lat) AS point,  -- 좌표
    s.name,  -- 매장 이름
    p.time,  -- 시간
    s.region,  -- 시/군/구
    p.region_cat,  --  시/군/구 카테고리
    p.m2024 + p.m2529 + p.w2024 + p.w2529 AS p20,  -- 20대 남녀 인구
FROM `[project_id].starbucks.starbucks_kr` AS s
JOIN `[project_id].starbucks.people_1225` AS p
ON s.region = p.region
WHERE s.city = "서울특별시" AND p.time = "2022-12-25 15:00:00"
)
```

!!! info "설정"

    === "fillColor"

        ![Image title](https://drive.google.com/uc?id=19sk0mFPKCWNlR4l5FOQs-un4TnqEUyxz){ class="skipglightbox" width=80% }

        각 카테고리는 서울시 자치구역 통합 계획에 따라 부여했다. 

        ![Image title](https://drive.google.com/uc?id=19zn8qnF8PgFe5iw_lbrbXu8lxFqX_L_x){ class="skipglightbox" align=left width=80% }

    === "fillOpacity"

        ![Image title](https://drive.google.com/uc?id=19sLrIOmItqEqU9CzQgLQVSV18JgIkKtp){ class="skipglightbox" width=100% }

    === "circleRadius"

        ![Image title](https://drive.google.com/uc?id=19sywjCFLpbUH6dyTq2Pi7wRkopqRz3Lz){ class="skipglightbox" width=80% }

### 최종 시각화 결과물

꼭 구글로 써야한다면 해당 툴을 쓰지만 조금 더 자유롭게 하려면 Python의 Folium 패키지 등 다른 툴을 쓰는 것이 좋아보인다.

![Image title](https://drive.google.com/uc?id=19YzCpliuViyYzOhOW99eCiXbbGKuP4wG){ class="skipglightbox" width=100% }