"""
Access Control Module
사용자 권한 관리 모듈
"""

from typing import Dict, List, Set
from langchain_core.documents import Document


class AccessControl:
    """문서 접근 제어 클래스"""

    def __init__(self):
        """
        초기화 - 사용자별 접근 권한 정의
        """
        # 사용자별 접근 가능한 문서 정의
        self.permissions: Dict[str, Set[str]] = {
            "admin": {
                "langchain_overview.md",
                "python_basics.md",
                "ai_ethics.md",
            },
            "developer": {
                "langchain_overview.md",
                "python_basics.md",
            },
            "guest": {
                "python_basics.md",
            },
        }

        # 사용자 역할별 설명
        self.role_descriptions = {
            "admin": "모든 문서 접근 가능",
            "developer": "기술 문서 접근 가능",
            "guest": "기본 문서만 접근 가능",
        }

    def can_access(self, user: str, document: str) -> bool:
        """
        사용자가 특정 문서에 접근할 수 있는지 확인

        Args:
            user: 사용자 이름
            document: 문서 이름 (예: "langchain_overview.md")

        Returns:
            bool: 접근 가능 여부
        """
        if user not in self.permissions:
            return False

        # 와일드카드 권한 확인 (admin은 모든 문서 접근 가능)
        if user == "admin" and "*" in self.permissions[user]:
            return True

        return document in self.permissions[user]

    def filter_documents(self, user: str, documents: List[Document]) -> List[Document]:
        """
        사용자 권한에 따라 문서 필터링

        Args:
            user: 사용자 이름
            documents: 문서 리스트

        Returns:
            List[Document]: 필터링된 문서 리스트
        """
        if user not in self.permissions:
            return []

        filtered = []
        for doc in documents:
            # 문서 이름 추출
            source = doc.metadata.get("source", "")
            if "/" in source:
                doc_name = source.split("/")[-1]
            else:
                doc_name = source

            # 권한 확인
            if self.can_access(user, doc_name):
                filtered.append(doc)

        return filtered

    def get_accessible_documents(self, user: str) -> List[str]:
        """
        사용자가 접근 가능한 문서 목록 반환

        Args:
            user: 사용자 이름

        Returns:
            List[str]: 접근 가능한 문서 이름 리스트
        """
        if user not in self.permissions:
            return []

        return sorted(list(self.permissions[user]))

    def add_permission(self, user: str, document: str):
        """
        사용자에게 문서 접근 권한 추가

        Args:
            user: 사용자 이름
            document: 문서 이름
        """
        if user not in self.permissions:
            self.permissions[user] = set()

        self.permissions[user].add(document)
        print(f"✅ {user}에게 {document} 접근 권한이 추가되었습니다.")

    def remove_permission(self, user: str, document: str):
        """
        사용자의 문서 접근 권한 제거

        Args:
            user: 사용자 이름
            document: 문서 이름
        """
        if user in self.permissions and document in self.permissions[user]:
            self.permissions[user].remove(document)
            print(f"✅ {user}의 {document} 접근 권한이 제거되었습니다.")
        else:
            print(f"⚠️  {user}는 {document}에 대한 권한이 없습니다.")

    def create_user(self, username: str, role: str = "guest"):
        """
        새 사용자 생성

        Args:
            username: 사용자 이름
            role: 역할 (admin/developer/guest)
        """
        if username in self.permissions:
            print(f"⚠️  사용자 {username}는 이미 존재합니다.")
            return

        # 역할에 따른 기본 권한 할당
        if role == "admin":
            self.permissions[username] = self.permissions["admin"].copy()
        elif role == "developer":
            self.permissions[username] = self.permissions["developer"].copy()
        else:
            self.permissions[username] = self.permissions["guest"].copy()

        print(f"✅ 사용자 {username} ({role}) 생성 완료")

    def get_user_info(self, username: str) -> dict:
        """
        사용자 정보 반환

        Args:
            username: 사용자 이름

        Returns:
            dict: 사용자 정보
        """
        if username not in self.permissions:
            return {"error": "사용자를 찾을 수 없습니다."}

        # 역할 추정
        role = "custom"
        if self.permissions[username] == self.permissions["admin"]:
            role = "admin"
        elif self.permissions[username] == self.permissions["developer"]:
            role = "developer"
        elif self.permissions[username] == self.permissions["guest"]:
            role = "guest"

        return {
            "username": username,
            "role": role,
            "accessible_documents": sorted(list(self.permissions[username])),
            "document_count": len(self.permissions[username]),
        }

    def list_all_users(self) -> List[dict]:
        """
        모든 사용자 정보 리스트 반환

        Returns:
            List[dict]: 사용자 정보 리스트
        """
        return [self.get_user_info(user) for user in self.permissions.keys()]


class RoleBasedAccessControl(AccessControl):
    """
    역할 기반 접근 제어 (RBAC)
    도전 과제용 - 더 세밀한 권한 관리
    """

    def __init__(self):
        super().__init__()

        # 역할 정의
        self.roles: Dict[str, Set[str]] = {
            "admin": {"read", "write", "delete", "manage_users"},
            "developer": {"read", "write"},
            "analyst": {"read"},
            "guest": {"read"},
        }

        # 사용자-역할 매핑
        self.user_roles: Dict[str, str] = {
            "admin": "admin",
            "developer": "developer",
            "guest": "guest",
        }

        # 문서별 필요 권한
        self.document_permissions: Dict[str, str] = {
            "langchain_overview.md": "read",
            "python_basics.md": "read",
            "ai_ethics.md": "read",
        }

    def can_access(self, user: str, document: str, action: str = "read") -> bool:
        """
        사용자가 특정 문서에 대한 특정 작업을 수행할 수 있는지 확인

        Args:
            user: 사용자 이름
            document: 문서 이름
            action: 작업 (read/write/delete)

        Returns:
            bool: 접근 가능 여부
        """
        # 사용자 역할 확인
        if user not in self.user_roles:
            return False

        user_role = self.user_roles[user]

        # 역할이 해당 작업 권한을 가지고 있는지 확인
        if action not in self.roles.get(user_role, set()):
            return False

        # 기본 문서 접근 권한 확인
        return super().can_access(user, document)

    def assign_role(self, user: str, role: str):
        """
        사용자에게 역할 할당

        Args:
            user: 사용자 이름
            role: 역할 (admin/developer/analyst/guest)
        """
        if role not in self.roles:
            print(f"❌ 유효하지 않은 역할: {role}")
            return

        self.user_roles[user] = role
        print(f"✅ {user}에게 {role} 역할이 할당되었습니다.")

        # 기본 문서 권한도 업데이트
        if role in self.permissions:
            self.permissions[user] = self.permissions[role].copy()

    def get_user_permissions(self, user: str) -> dict:
        """
        사용자의 상세 권한 정보 반환

        Args:
            user: 사용자 이름

        Returns:
            dict: 권한 정보
        """
        if user not in self.user_roles:
            return {"error": "사용자를 찾을 수 없습니다."}

        role = self.user_roles[user]
        actions = list(self.roles.get(role, set()))
        accessible_docs = self.get_accessible_documents(user)

        return {
            "username": user,
            "role": role,
            "actions": actions,
            "accessible_documents": accessible_docs,
        }
