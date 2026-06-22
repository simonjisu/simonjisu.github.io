# Soopace

Astro 기반 개인 연구 포트폴리오 및 블로그입니다.

## Requirements

- Node.js `>=22.12.0`
- 이 저장소의 권장 Node 버전은 `.nvmrc`에 적혀 있습니다.

`nvm`이 없는 환경에서는 `node -v`로 현재 버전을 확인하고, Node 22.12 이상을 설치한 뒤 진행하세요.

## Development

의존성 설치:

```bash
npm install
```

개발 서버 실행:

```bash
npm run dev
```

기본 주소:

```text
http://localhost:4321/
```

이미 `4321` 포트가 사용 중이면 Astro가 다음 포트로 자동 실행합니다. 예를 들어 `4322`로 뜨면 아래 주소로 접속하면 됩니다.

```text
http://localhost:4322/
```

## Build

정적 사이트 빌드:

```bash
npm run build
```

빌드 결과는 `dist/`에 생성됩니다.

## Research Page Toggle

연구 페이지 노출 여부는 아래 파일에서 조정합니다.

```text
src/research.config.yaml
```

예:

```yaml
projects:
  trace2map: true
  nl2sql-plus: false
```
