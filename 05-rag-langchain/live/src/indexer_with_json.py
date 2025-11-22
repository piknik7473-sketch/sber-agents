import json
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from config import config

logger = logging.getLogger(__name__)

def load_pdf_documents(data_dir: str) -> list:
    """Загрузка всех PDF документов из директории"""
    pages = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory {data_dir} does not exist")
        return pages
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages.extend(loader.load())
        logger.info(f"Loaded {pdf_file.name}")
    
    return pages

def split_documents(pages: list) -> list:
    """Разбиение документов на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов"""
    logger.info("Starting full reindexing...")
    
    try:
        # Загружаем PDF документы
        pages = load_pdf_documents(config.DATA_DIR)
        
        # Загружаем JSON документы
        data_path = Path(config.DATA_DIR)
        json_files = list(data_path.glob("*.json"))
        for json_file in json_files:
            json_docs = load_json_documents(str(json_file))
            pages.extend(json_docs)
        
        if not pages:
            logger.warning("No documents found to index")
            return None
        
        chunks = split_documents(pages)
        if not chunks:
            logger.warning("No chunks created after splitting")
            return None
            
        vector_store = create_vector_store(chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

def load_json_documents(json_file_path: str) -> list:
    """
    Загрузка документов из JSON файла с вопросами-ответами
    Каждая пара Q&A становится отдельным чанком
    """
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    try:
        # Загружаем JSON файл
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Создаем документы LangChain из каждого элемента массива
        documents = []
        for item in data:
            if 'full_text' in item:
                # Создаем Document с текстом из поля full_text
                doc = Document(
                    page_content=item['full_text'],
                    metadata={
                        'source': str(json_path),
                        'question': item.get('question', ''),
                        'category': item.get('category', ''),
                        'type': item.get('type', ''),
                        'url': item.get('url', '')
                    }
                )
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
        return documents
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {json_file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading JSON file {json_file_path}: {e}", exc_info=True)
        return []

