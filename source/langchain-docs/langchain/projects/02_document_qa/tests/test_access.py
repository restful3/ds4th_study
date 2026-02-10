"""
Tests for Access Control
"""

import pytest
from langchain_core.documents import Document
from access_control import AccessControl, RoleBasedAccessControl


@pytest.fixture
def access_control():
    """AccessControl 인스턴스 생성"""
    return AccessControl()


@pytest.fixture
def sample_documents():
    """테스트용 샘플 문서"""
    return [
        Document(
            page_content="LangChain content",
            metadata={"source": "/path/to/langchain_overview.md"},
        ),
        Document(
            page_content="Python content",
            metadata={"source": "/path/to/python_basics.md"},
        ),
        Document(
            page_content="AI Ethics content",
            metadata={"source": "/path/to/ai_ethics.md"},
        ),
    ]


def test_admin_access(access_control):
    """관리자 접근 권한 테스트"""
    assert access_control.can_access("admin", "langchain_overview.md")
    assert access_control.can_access("admin", "python_basics.md")
    assert access_control.can_access("admin", "ai_ethics.md")


def test_developer_access(access_control):
    """개발자 접근 권한 테스트"""
    assert access_control.can_access("developer", "langchain_overview.md")
    assert access_control.can_access("developer", "python_basics.md")
    assert not access_control.can_access("developer", "ai_ethics.md")


def test_guest_access(access_control):
    """게스트 접근 권한 테스트"""
    assert not access_control.can_access("guest", "langchain_overview.md")
    assert access_control.can_access("guest", "python_basics.md")
    assert not access_control.can_access("guest", "ai_ethics.md")


def test_unknown_user_access(access_control):
    """알 수 없는 사용자 접근 테스트"""
    assert not access_control.can_access("unknown", "langchain_overview.md")


def test_filter_documents_admin(access_control, sample_documents):
    """관리자 문서 필터링 테스트"""
    filtered = access_control.filter_documents("admin", sample_documents)
    assert len(filtered) == 3  # 모든 문서 접근 가능


def test_filter_documents_developer(access_control, sample_documents):
    """개발자 문서 필터링 테스트"""
    filtered = access_control.filter_documents("developer", sample_documents)
    assert len(filtered) == 2  # langchain, python만 접근 가능


def test_filter_documents_guest(access_control, sample_documents):
    """게스트 문서 필터링 테스트"""
    filtered = access_control.filter_documents("guest", sample_documents)
    assert len(filtered) == 1  # python만 접근 가능


def test_get_accessible_documents(access_control):
    """접근 가능 문서 목록 테스트"""
    admin_docs = access_control.get_accessible_documents("admin")
    assert len(admin_docs) == 3

    developer_docs = access_control.get_accessible_documents("developer")
    assert len(developer_docs) == 2

    guest_docs = access_control.get_accessible_documents("guest")
    assert len(guest_docs) == 1


def test_add_permission(access_control):
    """권한 추가 테스트"""
    access_control.add_permission("guest", "ai_ethics.md")
    assert access_control.can_access("guest", "ai_ethics.md")


def test_remove_permission(access_control):
    """권한 제거 테스트"""
    access_control.remove_permission("admin", "ai_ethics.md")
    assert not access_control.can_access("admin", "ai_ethics.md")


def test_create_user(access_control):
    """사용자 생성 테스트"""
    access_control.create_user("newuser", "developer")
    assert "newuser" in access_control.permissions
    assert access_control.can_access("newuser", "langchain_overview.md")


def test_get_user_info(access_control):
    """사용자 정보 조회 테스트"""
    info = access_control.get_user_info("admin")

    assert info["username"] == "admin"
    assert info["role"] == "admin"
    assert len(info["accessible_documents"]) == 3
    assert info["document_count"] == 3


def test_list_all_users(access_control):
    """모든 사용자 목록 테스트"""
    users = access_control.list_all_users()

    assert len(users) >= 3
    assert any(u["username"] == "admin" for u in users)
    assert any(u["username"] == "developer" for u in users)
    assert any(u["username"] == "guest" for u in users)


# RBAC Tests
@pytest.fixture
def rbac():
    """RoleBasedAccessControl 인스턴스 생성"""
    return RoleBasedAccessControl()


def test_rbac_read_access(rbac):
    """RBAC 읽기 권한 테스트"""
    assert rbac.can_access("admin", "langchain_overview.md", "read")
    assert rbac.can_access("developer", "python_basics.md", "read")
    assert rbac.can_access("guest", "python_basics.md", "read")


def test_rbac_write_access(rbac):
    """RBAC 쓰기 권한 테스트"""
    assert rbac.can_access("admin", "langchain_overview.md", "write")
    assert rbac.can_access("developer", "python_basics.md", "write")
    assert not rbac.can_access("guest", "python_basics.md", "write")


def test_rbac_delete_access(rbac):
    """RBAC 삭제 권한 테스트"""
    assert rbac.can_access("admin", "langchain_overview.md", "delete")
    assert not rbac.can_access("developer", "python_basics.md", "delete")
    assert not rbac.can_access("guest", "python_basics.md", "delete")


def test_rbac_assign_role(rbac):
    """RBAC 역할 할당 테스트"""
    rbac.assign_role("newuser", "developer")
    assert rbac.user_roles["newuser"] == "developer"


def test_rbac_get_user_permissions(rbac):
    """RBAC 사용자 권한 정보 조회 테스트"""
    perms = rbac.get_user_permissions("admin")

    assert perms["username"] == "admin"
    assert perms["role"] == "admin"
    assert "read" in perms["actions"]
    assert "write" in perms["actions"]
    assert "delete" in perms["actions"]
