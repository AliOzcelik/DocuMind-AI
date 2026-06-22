from datetime import datetime, timezone

from sqlalchemy import create_engine, ForeignKey, JSON, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker



engine = create_engine("sqlite:///chats.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass


def now_utc():
    return datetime.now(timezone.utc)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)
    messages: Mapped[list["Message"]] = relationship(back_populates="chat", cascade="all, delete-orphan", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String, nullable=True)
    sources: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    
    
def init_db():
    Base.metadata.create_all(bind=engine)
    
def ensure_chat(db, chat_id, title):
    if db.get(Chat, chat_id) is None:
        chat = Chat(id=chat_id, title=title)
        db.add(chat)
        db.commit()
        db.refresh(chat)
    
        
def add_message(db, chat_id, role, content, route=None, sources=None):
    message = Message(chat_id=chat_id, role=role, content=content, route=route, sources=sources)
    db.add(message)
    chat = db.get(Chat, chat_id)
    if chat:
        chat.updated_at = now_utc()
    db.commit()
    
    
def list_chats(db):
    #return db.query(Chat).order_by(Chat.created_at.desc()).all()
    return db.query(Chat).order_by(Chat.updated_at.desc()).all()

def get_messages(db, chat_id):
    #return db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return db.query(Message).filter_by(chat_id=chat_id).order_by(Message.id).all()

def delete_chat(db, chat_id):
    chat = db.get(Chat, chat_id)
    if chat:
        db.delete(chat)        # cascade deletes its messages (relationship cascade)
        db.commit()