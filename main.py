from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Form
from fastapi.responses import StreamingResponse, HTMLResponse
import os
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK SC"]
import io
from pymongo import MongoClient, errors, ReturnDocument
from datetime import datetime
import jieba  # библиотека сегментации китайского текста на слова

app = FastAPI(title="Text Comparison API with Web UI", version="2.3.0")

UPLOAD_DIR = "uploads"
MODEL_DIR = os.environ.get("MODEL_PATH", "/app/model")
HF_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Глобальные переменные для подключения к MongoDB
client = None
db = None
comparisons_collection = None

# Глобальная переменная для модели эмбеддингов
_embedding_model = None


def preprocess_for_tfidf(text: str) -> str:
    """
    Подготовка текста перед TF-IDF-векторизацией.

    Проблема: в китайском нет пробелов между словами, поэтому стандартный
    токенизатор TfidfVectorizer (режет по пробелам) видит весь абзац как
    один «токен» и сходство получается бессмысленным.

    Решение: если в тексте есть китайские иероглифы — режем его на слова
    через jieba и разделяем пробелами. После этого TF-IDF работает со
    словами, как с обычным текстом.

    Для русского и английского текст не меняется — им сегментация не нужна.
    """
    # Диапазон \u4e00–\u9fff — стандартные китайские иероглифы Unicode (CJK Unified Ideographs).
    # any(...) проверяет: есть ли хоть один такой символ в тексте.
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        # jieba.cut(text) возвращает список слов/морфем:
        # например, "春节是中国最重要的节日" -> ["春节", "是", "中国", "最", "重要", "的", "节日"]
        # " ".join(...) склеивает их пробелами — формат, который понимает TfidfVectorizer.
        return " ".join(jieba.cut(text))
    # Ни одного иероглифа — возвращаем текст без изменений.
    return text


def get_mongo_collection():
    """Ленивое подключение к MongoDB с проверкой доступности."""
    global client, db, comparisons_collection
    if comparisons_collection is not None:
        return comparisons_collection

    try:
        client = MongoClient(
            "mongodb://root:password@mongo:27017/?authSource=admin",
            serverSelectionTimeoutMS=2000
        )
        client.admin.command('ping')
        db = client["fastapi_files"]
        comparisons_collection = db["comparisons"]
        return comparisons_collection
    except errors.ServerSelectionTimeoutError:
        print("MongoDB not reachable; reports will not be saved.")
        return None
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return None


def get_embedding_model():
    """Ленивая загрузка модели эмбеддингов sentence-transformers."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        # Приоритет: локальная копия из volume, иначе скачивание с Hugging Face
        if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
            source = MODEL_DIR
        else:
            source = HF_MODEL_NAME
        print(f"Loading embeddings model from {source}...")
        _embedding_model = SentenceTransformer(source)
        print("Embeddings model loaded.")
        return _embedding_model
    except Exception as e:
        print(f"Failed to load embeddings model: {e}")
        return None



def get_next_number() -> Optional[int]:
    """Атомарно увеличивает счётчик и возвращает новый порядковый номер."""
    collection = get_mongo_collection()
    if collection is None:
        return None

    try:
        result = collection.database["counters"].find_one_and_update(
            {"_id": "file_number"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        return result["seq"]
    except Exception as e:
        print(f"Failed to get next number: {e}")
        return None


# ---------------------------------------------------------
# Вспомогательные функции для работы с коллекцией files
# ---------------------------------------------------------

def save_file_record(filename: str, size_bytes: int):
    """Сохраняет учётную карточку файла с порядковым номером."""
    collection = get_mongo_collection()
    if collection is None:
        return None, None

    number = get_next_number()
    files_collection = collection.database["files"]

    record = {
        "filename": filename,
        "size_bytes": size_bytes,
        "uploaded_at": datetime.utcnow(),
        "status": "uploaded",
        "number": number
    }

    try:
        result = files_collection.insert_one(record)
        return str(result.inserted_id), number
    except Exception as e:
        print(f"Failed to save file record: {e}")
        return None, None


def parse_numbers(raw: str) -> List[int]:
    """Разбирает строку '1, 3, 2' в список целых чисел с валидацией."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    if not parts:
        raise HTTPException(status_code=400, detail="Empty numbers list")

    nums = []
    for p in parts:
        if not p.isdigit():
            raise HTTPException(status_code=400, detail=f"Invalid number: '{p}'")
        nums.append(int(p))

    if len(nums) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 file numbers separated by commas")

    if len(set(nums)) != len(nums):
        raise HTTPException(status_code=400, detail="Numbers must be unique")

    return nums


def get_files_by_numbers(numbers: List[int]):
    """Возвращает документы файлов по номерам в заданном порядке (включая архив)."""
    collection = get_mongo_collection()
    if collection is None:
        return None, "MongoDB not reachable"

    files_collection = collection.database["files"]
    docs = {d["number"]: d for d in files_collection.find({"number": {"$in": numbers}})}

    missing = [n for n in numbers if n not in docs]
    if missing:
        return None, f"Files with numbers not found: {', '.join(map(str, missing))}"

    return [docs[n] for n in numbers], None


def read_files_contents(docs: List[dict]) -> List[str]:
    """Читает содержимое файлов с диска в порядке переданных документов."""
    contents = []
    for d in docs:
        path = os.path.join(UPLOAD_DIR, d["filename"])
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {d['filename']}")
        with open(path, "r", encoding="utf-8") as f:
            contents.append(f.read())
    return contents


def files_meta(docs: List[dict]) -> List[dict]:
    """Краткое описание файлов для отчётов и ответов."""
    return [
        {"number": d["number"], "filename": d["filename"], "file_id": str(d["_id"])}
        for d in docs
    ]


# ---------------------------------------------------------
# Методы вычисления схожести
# ---------------------------------------------------------

def compute_tfidf_similarity(contents: List[str]) -> np.ndarray:
    """Метод 1: TF-IDF + косинусное сходство (лексическая близость)."""
    # Сегментация китайского: без неё векторайзер видит весь текст без пробелов
    # как один токен и сходство получается бессмысленным.
    contents = [preprocess_for_tfidf(t) for t in contents]
    # Приведение к нижнему регистру — для английских слов (Chinese/CHINESE и т. п.);
    # у иероглифов регистра нет, так что им это не мешает.
    contents = [t.lower() for t in contents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    return cosine_similarity(tfidf_matrix)



def compute_embeddings_similarity(contents: List[str]):
    """Метод 2: эмбеддинги sentence-transformers + косинусное сходство (семантическая близость).
    Возвращает матрицу или None, если модель недоступна."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        embeddings = model.encode(contents, normalize_embeddings=True)
        embeddings = np.array(embeddings)
        return np.dot(embeddings, embeddings.T)
    except Exception as e:
        print(f"Embeddings computation failed: {e}")
        return None


def off_diagonal_average(matrix: np.ndarray) -> float:
    """Средняя схожесть по всем парам вне диагонали."""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    return round(float(matrix[np.triu_indices(n, k=1)].mean()), 3)


# ---------------------------------------------------------
# Эндпоинты API
# ---------------------------------------------------------

@app.post("/documents/")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    file_id, number = save_file_record(file.filename, len(contents))

    response = {
        "filename": file.filename,
        "path": file_path,
        "message": "File saved to disk",
        "file_id": file_id,
        "number": number
    }

    if file_id is None:
        response["warning"] = "File saved on disk, but record not saved to MongoDB (check logs)"

    return response


@app.get("/documents/{number}")
async def get_document_content(number: int):
    """Возвращает содержимое файла по его порядковому номеру."""
    collection = get_mongo_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not reachable")

    doc = collection.database["files"].find_one({"number": number})
    if not doc:
        raise HTTPException(status_code=404, detail=f"File with number {number} not found")

    path = os.path.join(UPLOAD_DIR, doc["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found on disk: {doc['filename']}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "number": number,
        "filename": doc["filename"],
        "status": doc.get("status"),
        "size_bytes": doc.get("size_bytes"),
        "uploaded_at": doc.get("uploaded_at").isoformat() if isinstance(doc.get("uploaded_at"), datetime) else None,
        "content": content
    }


@app.post("/documents/{number}/archive")
async def archive_document(number: int):
    """Убирает файл в архив: меняет status на 'archived', сам документ не удаляется."""
    collection = get_mongo_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not reachable")

    result = collection.database["files"].update_one(
        {"number": number},
        {"$set": {"status": "archived", "archived_at": datetime.utcnow()}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"File with number {number} not found")

    return {"number": number, "status": "archived", "message": "File moved to archive"}


@app.post("/documents/{number}/restore")
async def restore_document(number: int):
    """Возвращает файл из архива: меняет status обратно на 'uploaded'."""
    collection = get_mongo_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not reachable")

    result = collection.database["files"].update_one(
        {"number": number},
        {"$set": {"status": "uploaded"}, "$unset": {"archived_at": ""}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"File with number {number} not found")

    return {"number": number, "status": "uploaded", "message": "File restored from archive"}


@app.post("/compare/")
async def compare_documents(numbers: str = Form(..., description="Номера файлов через запятую, например: 1,3,2")):
    nums = parse_numbers(numbers)

    docs, err = get_files_by_numbers(nums)
    if err:
        raise HTTPException(status_code=404, detail=err)

    contents = read_files_contents(docs)

    if any(not c.strip() for c in contents):
        raise HTTPException(status_code=400, detail="Files cannot be empty")

    meta = files_meta(docs)

    # Метод 1: TF-IDF — считается всегда
    sim_tfidf = compute_tfidf_similarity(contents)
    tfidf_list = [[round(x, 3) for x in row] for row in sim_tfidf.tolist()]
    avg_tfidf = off_diagonal_average(sim_tfidf)

    # Метод 2: эмбеддинги — считаются, если модель доступна
    sim_emb = compute_embeddings_similarity(contents)
    emb_list = None
    avg_emb = None
    if sim_emb is not None:
        emb_list = [[round(x, 3) for x in row] for row in sim_emb.tolist()]
        avg_emb = off_diagonal_average(sim_emb)

    collection = get_mongo_collection()
    saved_to_mongodb = False
    if collection is not None:
        report = {
            "files": meta,
            "numbers": nums,
            "average_score_tfidf": avg_tfidf,
            "average_score_embeddings": avg_emb,
            "matrix_tfidf": tfidf_list,
            "matrix_embeddings": emb_list,
            "timestamp": datetime.utcnow(),
            "status": "completed"
        }
        try:
            collection.insert_one(report)
            saved_to_mongodb = True
        except Exception as e:
            print(f"Failed to save report to MongoDB: {e}")

    response = {
        "files": meta,
        "similarity_matrix_tfidf": tfidf_list,
        "average_score_tfidf": avg_tfidf,
        "saved_to_mongodb": saved_to_mongodb
    }

    if emb_list is not None:
        response["similarity_matrix_embeddings"] = emb_list
        response["average_score_embeddings"] = avg_emb
    else:
        response["warning"] = "Embeddings model not available (check MODEL_DIR and logs)"

    return response


@app.post("/heatmap/")
async def generate_heatmap(
    numbers: str = Form(..., description="Номера файлов через запятую, например: 1,3,2"),
    method: str = Form("tfidf", description="Метод сравнения: tfidf или embeddings")
):
    if method not in ("tfidf", "embeddings"):
        raise HTTPException(status_code=400, detail="method must be 'tfidf' or 'embeddings'")

    nums = parse_numbers(numbers)

    docs, err = get_files_by_numbers(nums)
    if err:
        raise HTTPException(status_code=404, detail=err)

    contents = read_files_contents(docs)

    if any(not c.strip() for c in contents):
        raise HTTPException(status_code=400, detail="Files cannot be empty")

    if method == "tfidf":
        similarity = compute_tfidf_similarity(contents)
        title = "Similarity matrix (TF-IDF)"
    else:
        similarity = compute_embeddings_similarity(contents)
        if similarity is None:
            raise HTTPException(
                status_code=503,
                detail="Embeddings model not available (check MODEL_DIR and logs)"
            )
        title = "Similarity matrix (Embeddings)"

    labels = [f"№{d['number']}:{d['filename'][:12]}" for d in docs]

    size = max(4, 0.8 * len(nums))
    plt.figure(figsize=(size, size))
    plt.imshow(similarity, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=1)
    plt.title(title)
    plt.xticks(range(len(nums)), labels, rotation=45, ha='right')
    plt.yticks(range(len(nums)), labels)
    plt.colorbar()

    for i in range(len(nums)):
        for j in range(len(nums)):
            plt.text(j, i, f"{similarity[i][j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/files/")
async def get_files(limit: int = 20, include_archived: bool = False):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not reachable", "files": []}

    files_collection = collection.database["files"]

    query = {} if include_archived else {"status": "uploaded"}
    cursor = files_collection.find(query, {"_id": 0}).sort("number", 1).limit(limit)
    files = list(cursor)

    for f in files:
        ts = f.get("uploaded_at")
        if ts and isinstance(ts, datetime):
            f["uploaded_at"] = ts.isoformat()
        ts = f.get("archived_at")
        if ts and isinstance(ts, datetime):
            f["archived_at"] = ts.isoformat()

    return {"files": files, "total": len(files)}


@app.get("/reports/")
async def get_reports(limit: int = 10):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not reachable", "reports": []}

    cursor = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    reports = list(cursor)

    for r in reports:
        ts = r.get("timestamp")
        if ts and isinstance(ts, datetime):
            r["timestamp"] = ts.isoformat()

    return {"reports": reports, "total": len(reports)}


# ---------------------------------------------------------
# Веб-интерфейс: таблица файлов, загрузка, просмотр, сравнение, архив
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>Сравнение текстов — FastAPI</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }
            h1 { margin-bottom: 0.5rem; }
            .hint { color: #666; font-size: 0.9rem; }
            .section { margin: 1.5rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
            table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
            th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }
            th { background: #f0f0f0; }
            button { margin-top: 0.75rem; padding: 0.5rem 1rem; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; }
            button.small { padding: 0.25rem 0.6rem; margin: 0 0.15rem; font-size: 0.8rem; }
            button.archive { background: #8a8a8a; }
            button.archive:hover { background: #6d6d6d; }
            button.restore { background: #2e8b57; }
            button.restore:hover { background: #246b44; }
            input[type=file] { margin-top: 0.5rem; }
            input[type=text] { padding: 0.4rem 0.6rem; border: 1px solid #ccc; border-radius: 4px; width: 220px; }
            select { padding: 0.4rem 0.6rem; border: 1px solid #ccc; border-radius: 4px; }
            label { font-weight: 500; }
            label.checkbox { font-weight: 400; }
            pre { background: #f4f4f4; padding: 0.5rem; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }
            .status-ok { color: #1a7f37; }
            .status-err { color: #c0392b; }
            .empty { color: #999; font-style: italic; }
            .tag-archived { color: #8a6d3b; background: #fcf8e3; padding: 0 0.35rem; border-radius: 3px; border: 1px solid #faebcc; }
        </style>
    </head>
    <body>
        <h1>Сравнение текстовых файлов</h1>

        <div class="section">
            <h3>Загруженные файлы</h3>
            <label class="checkbox"><input type="checkbox" id="showArchived"> Показать с архивом</label>
            <table id="files-table">
                <thead>
                    <tr>
                        <th>№</th>
                        <th>Имя файла</th>
                        <th>Статус</th>
                        <th>Размер (байт)</th>
                        <th>Загружен</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody id="files-table-body">
                    <tr><td colspan="6" class="empty">Загрузка...</td></tr>
                </tbody>
            </table>
            <div id="upload-status"></div>
            <label>Новый файл (.txt):</label>
            <input type="file" id="fileInput" accept=".txt">
            <button id="uploadBtn">Загрузить файл</button>
        </div>

        <div class="section" id="view-section" style="display:none">
            <h3>Просмотр файла: <span id="view-title"></span></h3>
            <pre id="view-content"></pre>
        </div>

        <div class="section">
            <h3>Сравнение файлов</h3>
            <label>Номера файлов через запятую (например: 1,3,2):</label>
            <input type="text" id="numbers" placeholder="1,3,2">
            <label style="margin-left:1rem">Метод:</label>
            <select id="method">
                <option value="tfidf">TF-IDF (лексическая близость)</option>
                <option value="embeddings">Эмбеддинги (семантическая близость)</option>
            </select>
            <button id="compareBtn">Сравнить файлы</button>
            <div id="compare-status"></div>
        </div>

        <div class="section" id="heatmap-section" style="display:none">
            <h3>Тепловая карта схожести</h3>
            <img id="heatmap-img" src="" alt="Тепловая карта схожести" style="max-width:100%">
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadStatus = document.getElementById('upload-status');
            const showArchived = document.getElementById('showArchived');
            const numbersInput = document.getElementById('numbers');
            const methodSelect = document.getElementById('method');
            const compareBtn = document.getElementById('compareBtn');
            const compareStatus = document.getElementById('compare-status');
            const viewSection = document.getElementById('view-section');
            const viewTitle = document.getElementById('view-title');
            const viewContent = document.getElementById('view-content');
            const heatmapSection = document.getElementById('heatmap-section');
            const heatmapImg = document.getElementById('heatmap-img');
            const filesBody = document.getElementById('files-table-body');

            function escapeHtml(s) {
                const div = document.createElement('div');
                div.textContent = s == null ? '' : String(s);
                return div.innerHTML;
            }

            function formatDateTime(iso) {
                if (!iso) return '—';
                return new Date(iso).toLocaleString();
            }

            async function loadFiles() {
                const url = '/files/' + (showArchived.checked ? '?include_archived=true' : '');
                try {
                    const res = await fetch(url);
                    const data = await res.json();
                    filesBody.innerHTML = '';
                    if (!data.files || data.files.length === 0) {
                        filesBody.innerHTML = '<tr><td colspan="6" class="empty">Пока нет загруженных файлов</td></tr>';
                        return;
                    }
                    data.files.forEach(f => {
                        const isArchived = f.status === 'archived';
                        const tr = document.createElement('tr');
                        const statusHtml = isArchived
                            ? '<span class="tag-archived">archived</span>'
                            : escapeHtml(f.status);
                        const buttons = isArchived
                            ? `<button class="small restore" data-action="restore" data-number="${f.number}">Вернуть из архива</button>
                               <button class="small" data-action="view" data-number="${f.number}">Просмотреть файл</button>`
                            : `<button class="small" data-action="view" data-number="${f.number}">Просмотреть файл</button>
                               <button class="small archive" data-action="archive" data-number="${f.number}">Убрать в архив</button>`;
                        tr.innerHTML = `
                            <td>${f.number != null ? f.number : '—'}</td>
                            <td>${escapeHtml(f.filename)}</td>
                            <td>${statusHtml}</td>
                            <td>${f.size_bytes != null ? f.size_bytes : '—'}</td>
                            <td>${formatDateTime(f.uploaded_at)}</td>
                            <td>${buttons}</td>
                        `;
                        filesBody.appendChild(tr);
                    });
                } catch (e) {
                    filesBody.innerHTML = '<tr><td colspan="6" class="empty">Не удалось загрузить список файлов</td></tr>';
                }
            }

            showArchived.addEventListener('change', loadFiles);

            filesBody.addEventListener('click', async (e) => {
                const btn = e.target.closest('button[data-number]');
                if (!btn) return;
                const number = btn.dataset.number;
                const action = btn.dataset.action;

                if (action === 'view') {
                    try {
                        const res = await fetch('/documents/' + number);
                        if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || 'Ошибка загрузки файла');
                        }
                        const doc = await res.json();
                        viewTitle.textContent = '№' + doc.number + ' — ' + doc.filename;
                        viewContent.textContent = doc.content || '(пустой файл)';
                        viewSection.style.display = 'block';
                        viewSection.scrollIntoView({ behavior: 'smooth' });
                    } catch (err) {
                        viewTitle.textContent = 'Ошибка';
                        viewContent.textContent = err.message;
                        viewSection.style.display = 'block';
                    }
                }

                if (action === 'archive' || action === 'restore') {
                    btn.disabled = true;
                    try {
                        const res = await fetch('/documents/' + number + '/' + action, { method: 'POST' });
                        if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || 'Ошибка операции');
                        }
                        loadFiles();
                    } catch (err) {
                        alert('Ошибка: ' + err.message);
                        btn.disabled = false;
                    }
                }
            });

            uploadBtn.addEventListener('click', async () => {
                const f = fileInput.files[0];
                uploadStatus.textContent = '';
                if (!f) {
                    uploadStatus.innerHTML = '<span class="status-err">Выберите файл.</span>';
                    return;
                }
                if (!f.name.toLowerCase().endsWith('.txt')) {
                    uploadStatus.innerHTML = '<span class="status-err">Разрешены только файлы .txt</span>';
                    return;
                }
                uploadBtn.disabled = true;
                uploadStatus.textContent = 'Загрузка...';

                const formData = new FormData();
                formData.append('file', f);

                try {
                    const res = await fetch('/documents/', { method: 'POST', body: formData });
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        throw new Error(err.detail || 'Ошибка загрузки');
                    }
                    const json = await res.json();
                    uploadStatus.innerHTML = '<span class="status-ok">Загружен файл №' + json.number + ' — ' + escapeHtml(json.filename) + '</span>';
                    fileInput.value = '';
                    loadFiles();
                } catch (e) {
                    uploadStatus.innerHTML = '<span class="status-err">Ошибка: ' + escapeHtml(e.message) + '</span>';
                } finally {
                    uploadBtn.disabled = false;
                }
            });

            compareBtn.addEventListener('click', async () => {
                const numbers = numbersInput.value.trim();
                const method = methodSelect.value;
                compareStatus.textContent = '';
                heatmapSection.style.display = 'none';

                if (!numbers) {
                    compareStatus.innerHTML = '<span class="status-err">Укажите номера файлов через запятую.</span>';
                    return;
                }

                compareBtn.disabled = true;
                compareStatus.textContent = 'Выполняется сравнение...';

                const formData = new FormData();
                formData.append('numbers', numbers);
                formData.append('method', method);

                try {
                    const res = await fetch('/heatmap/', { method: 'POST', body: formData });
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        throw new Error(err.detail || 'Ошибка сравнения');
                    }
                    const blob = await res.blob();
                    heatmapImg.src = URL.createObjectURL(blob);
                    heatmapSection.style.display = 'block';
                    heatmapSection.scrollIntoView({ behavior: 'smooth' });
                    compareStatus.innerHTML = '<span class="status-ok">Готово (' + method + ')</span>';
                } catch (e) {
                    compareStatus.innerHTML = '<span class="status-err">Ошибка: ' + escapeHtml(e.message) + '</span>';
                } finally {
                    compareBtn.disabled = false;
                }
            });

            loadFiles();
        </script>
    </body>
    </html>
    """
    return html
