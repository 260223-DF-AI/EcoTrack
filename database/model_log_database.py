from sqlalchemy import Integer, Numeric, String, Boolean, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pandas import DataFrame
from dotenv import load_dotenv
from os import getenv

class Base(DeclarativeBase):
    pass

class AuditLog(Base):
    """Object for the auditlog table in postgreSQL"""
    __tablename__ = 'auditlog'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    species: Mapped[str] = mapped_column(String(25), nullable=True)
    all_statuses: Mapped[str] = mapped_column(Text, nullable=True)
    endangered_status: Mapped[str] = mapped_column(String(30), nullable=True)
    classifier_confidence: Mapped[float] = mapped_column(Numeric(5,2), nullable=True)
    unusual_location: Mapped[bool] = mapped_column(Boolean, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    llm_confidence: Mapped[float] = mapped_column(Numeric(5,2), nullable=True)



class Database:
    def __init__(self):
        """Load the postgreSQL database connection string and initialize the engine"""
        load_dotenv()

        self.CS = getenv('CS')
        self.engine = create_engine(self.CS)
        Base.metadata.create_all(self.engine, checkfirst=True)

    def add_log(self, classifier_response: dict, llm_response: dict) -> None:
        """Records the outputs from both models to the audit logs

        Args:
            classifier_response - Result provided by the image classification model
            llm_response - Result returned from gemini if an endangered species was identified. Otherwise all fields are None
        """
        responses = {**classifier_response, **llm_response} # combine into one dictionary, the classifier response one

        # round confidence scores to the 100ths decimal place
        responses['classifier_confidence'] = round(responses['classifier_confidence'], 2)
        if responses['llm_confidence'] is not None:
            responses['llm_confidence'] = round(responses['llm_confidence'], 2)
        df = DataFrame([responses]) # convert to pandas DataFrame so it can easily be sent to sql
        df.to_sql(name='auditlog', con=self.engine, index=False, if_exists='append')




