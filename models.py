from sqlalchemy import Column, Integer, Text
from database import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text)
    content = Column(Text)
