# GitHub Pages `/docs` 전환 전 점검

현재 Pages는 저장소 루트를 공개하므로 `archive/` 아래의 기존 HTML URL도 노출된다. Pages 소스를 `main/docs`로 바꾸면 이 URL들은 자동으로 사라진다.

전환 전에 다음을 완료한다.

1. `agent-support/legacy-pages.toml`의 모든 경로를 확인한다.
2. 각 항목을 `preserve`, `redirect`, `retire` 중 하나로 결정한다.
3. 보존할 URL은 기존 URL과 같은 `docs/archive/...` 경로에 정적 파일 또는 리다이렉트를 만든다.
4. 연결된 이미지·CSS·JavaScript가 필요한 경우 함께 보존한다.
5. 저작권 또는 개인정보 문제가 있는 항목은 공개 보존하지 않는다.
6. 로컬 서버에서 홈페이지, 새 발표 URL과 보존 대상 기존 URL을 확인한다.
7. 검증 CI가 통과한 뒤 운영자가 GitHub Pages 소스를 한 번만 `main /docs`로 변경한다.

GitHub Actions는 검증 전용으로 유지한다. Pages 배포 방식과 혼용하지 않는다.
