"""
Seed script: creates a default admin user for development.
Run: python -m app.scripts.seed_admin
"""

import uuid

from app.core.security import get_password_hash
from app.db.base import Base
from app.db.models import *  # noqa - ensure all models loaded
from app.db.models.user import Role, User, UserRole
from app.db.snowflake import SessionLocal, engine

# Create tables
Base.metadata.create_all(bind=engine)


def seed():
    db = SessionLocal()
    try:
        # Check if admin already exists
        existing = db.query(User).filter(User.email == "admin@user.local").first()
        if existing:
            print(f"Admin user already exists: {existing.email} (id={existing.id})")
            return

        # Create roles
        for role_name, role_desc in [
            ("admin", "Full system access"),
            ("analyst", "Client management and queries"),
            ("viewer", "Read-only access"),
        ]:
            existing_role = db.query(Role).filter(Role.name == role_name).first()
            if not existing_role:
                db.add(
                    Role(id=str(uuid.uuid4()), name=role_name, description=role_desc)
                )
                print(f"Created role: {role_name}")

        db.commit()

        # Create admin user
        admin_id = str(uuid.uuid4())
        admin = User(
            id=admin_id,
            email="admin@user.local",
            password_hash=get_password_hash("admin123"),
            full_name="Admin User",
            is_active=True,
        )
        db.add(admin)
        db.commit()

        # Assign admin role
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if admin_role:
            db.add(UserRole(user_id=admin_id, role_id=admin_role.id))
            db.commit()

        print(f"Created admin user: admin@user.local / admin123 (id={admin_id})")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
