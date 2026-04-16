from sqlalchemy import Integer, Numeric, String, Boolean, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pandas import DataFrame
from dotenv import load_dotenv
from os import getenv

class Base(DeclarativeBase):
    pass

class AuditLog(Base):
    __tablename__ = 'auditlog'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    species: Mapped[str] = mapped_column(String(25))
    endangered_status: Mapped[str] = mapped_column(String(30))
    classifier_confidence: Mapped[float] = mapped_column(Numeric(5,2))
    unusual_location: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[str] = mapped_column(Text)
    llm_confidence: Mapped[float] = mapped_column(Numeric(5,2))



class Database:
    def __init__(self):
        load_dotenv()

        self.CS = getenv('CS')
        self.engine = create_engine(self.CS)

    def add_log(self, classifier_response: dict, llm_response: dict):
        responses = classifier_response.update(llm_response)
        df = DataFrame(responses)
        df.to_sql(name='auditlog', con=self.CS, index=False, if_exists='append')




