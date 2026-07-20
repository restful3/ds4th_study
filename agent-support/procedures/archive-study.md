# 완료된 스터디 아카이브 절차

이 절차는 사용자가 특정 스터디의 종료와 이동을 명시적으로 요청했을 때만 수행한다.

1. `git status --short`를 확인하고 대상 경로와 겹치는 기존 변경이 있으면 먼저 보고한다.
2. `agent-support/studies.toml`에서 대상 스터디의 `materials_path`, `archive_path`, `presentation_path`를 확인한다.
3. 대상이 `active`인지, 원본 경로가 존재하고 archive 대상 경로가 비어 있는지 확인한다.
4. 학습자료만 `git mv <materials_path> <archive_path>`로 이동한다.
5. 레지스트리의 `status`를 `archived`로 바꾸고 `materials_path`를 `archive_path`와 같게 갱신한다.
6. `docs/studies/<study-slug>`와 그 아래 발표 폴더는 이동하거나 이름을 바꾸지 않는다.
7. `README.md`의 진행 중 일정과 아카이브 링크를 실제 상태에 맞게 갱신한다.
8. 인덱스를 다시 생성하고 사이트 및 저장소 링크를 검증한다.
9. 이동된 파일 수와 공개 URL이 유지됨을 보고한다. 커밋과 푸시는 별도 명시 요청이 있을 때만 수행한다.

과거 자료가 이미 `archive/` 밖에 있는 등 레거시 상태가 다르면 자동으로 대규모 정리하지 말고 예외를 보고한다.
