import os
import uuid
import json
import base64
import tiktoken
import re
import requests
from pdf2image import convert_from_path
from openai import OpenAI
from neo4j import GraphDatabase
from io import BytesIO
from pymilvus import MilvusClient, DataType
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from modelscope import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader, PdfWriter
import tempfile
from PIL import Image

NEO4J_URI = "neo4j://localhost://7474"
NEO4J_AUTH = ("neo4j", "12345678")
MAX_TOKENS = 256
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = 'default'
SILICONFLOW_API_KEY = 'sk-lfgwvzyqxmwxtomwdqqjbdlvibfdrravglobjhiuvnqnfwyx'

template_for_tablename_identify = """
你是一个表格分析专家。你将看到一张图片和该图片中某一个表格的OCR识别结果。
请你结合图像内容和OCR识别结果，从图片中识别该表格的可能性最大的表名。表名通常位于表格上方，可能以“表x.x”、标题文字或其他描述形式出现。
OCR识别结果如下：
{}
最后按照以下格式输出：
```json
{{
    "name": ""
}}
"""

template_for_table_ = """
你是一个表格分析专家。你将看到一张含有两个表格的图片。
请遵从以下规则判断这两个表格是同一个表格的不同部分，还是两个独立的表格。
1. 列名是否一致
2. 列数，列宽是否一致
3. 序号是否连贯
4. 内容是否有衔接性
最后按照以下格式输出，1表示是同一个表格的不同部分，0表示是独立的表格：
```json
{{
    "type": 1
}}
"""

def image_to_base64_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_base64_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def pdf_to_base64_images_by_batch(pdf_path, start_page=0, end_page=None, batch_size=30):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if end_page is None:
        end_page = total_pages
    base64_images = []

    for i in range(start_page, end_page, batch_size):
        batch_writer = PdfWriter()

        for j in range(i, min(i + batch_size, end_page)):
            batch_writer.add_page(reader.pages[j])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            batch_writer.write(tmp_pdf)
            tmp_pdf_path = tmp_pdf.name

        images = convert_from_path(tmp_pdf_path,dpi=300)
        for img in images:
            base64_images.append(image_to_base64_from_pil(img))

        print(f"已转换 {min(i + batch_size, end_page)} / {end_page} 页")

        os.remove(tmp_pdf_path)

    return base64_images

def vl_chat_from_path(image_path, prompt):
    client = OpenAI(
        api_key="sk-9536a97947b641ad9e287da238ba3abb",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[
            {
                "role": "user","content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64_from_path(image_path)}"}
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

def vl_chat(base64_image, prompt):
    client = OpenAI(
        api_key="sk-9536a97947b641ad9e287da238ba3abb",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[
            {
                "role": "user","content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

def parse_json(content: str):
    try:
        start = content.rfind("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_string = content[start: end+1]
            json_data = json.loads(s=json_string)
            return json_data
        else:
            raise Exception('parse json error!\n')
    except Exception as e:
        print(e)

def clear_graph(url: str, auth: tuple):
    with GraphDatabase.driver(url, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("""
            MATCH (p)
            DETACH DELETE p
            """,
            database_="neo4j",
        )

def create_nodes(url: str, auth: tuple, table_data: list):
    with GraphDatabase.driver(url, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("""
            UNWIND $table_data AS table
            CREATE (n:Table {
                table_id: table.table_id, 
                table_caption: table.table_caption,
                page_idx: table.page_idx,
                pdf_path: table.pdf_path
            })
            """,
            table_data=table_data,
            database_="neo4j",
        )

def create_relationship(url: str, auth: tuple, table_id_1: str, table_id_2: str):
    with GraphDatabase.driver(url, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("""
            MATCH(a:Table {table_id: $table_id_1}), (b:Table {table_id: $table_id_2})
            CREATE (a)-[:NEXT]->(b)
            """,
            table_id_1=table_id_1,
            table_id_2=table_id_2,
            database_="neo4j",
        )

def search_nodes(url: str, auth: tuple, table_id: str):
    with GraphDatabase.driver(url, auth=auth) as driver:
        driver.verify_connectivity()
        result = driver.execute_query("""
            MATCH (n:Table {table_id: $table_id})
            OPTIONAL MATCH (n)-[:NEXT*1..]-(related)
            WITH collect(DISTINCT n) + collect(DISTINCT related) AS allTables
            UNWIND allTables AS t
            ORDER BY size(t.table_caption) > 0 DESC, t.page_idx ASC
            RETURN t
            """,
            table_id=table_id,
            database_="neo4j",
        )
    return result

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def html_to_rows(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True).replace("\n", " ") for td in tr.find_all("td")]
        rows.append(cells)
    return rows

def format_markdown_row(cells):
    return "| " + " | ".join(cells) + " |"

def split_large_cell(cell):
    parts = re.split(r'(?<=[；。])|(?<=\d\）)', cell)
    parts = [p.strip() for p in parts if p.strip()]
    
    chunks = []
    current = []
    
    for part in parts:
        current.append(part)
        token_count = count_tokens(" ".join(current))
        if token_count > MAX_TOKENS:
            current.pop()
            if current:
                chunks.append("".join(current))
            # 如果单个 part 超过 256，则继续 fallback 为 word-level
            if count_tokens(part) > MAX_TOKENS:
                chunks.extend(_fallback_split_by_word(part))
            else:
                current = [part]
        else:
            continue

    if current:
        chunks.append("".join(current))

    return chunks

def _fallback_split_by_word(text):
    """当句段拆分后仍超限时,按 word 切分"""
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) > MAX_TOKENS:
            current.pop()
            chunks.append(" ".join(current))
            current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_rows(rows):
    chunks = []
    i = 0
    while i < len(rows):
        group = []
        token_total = 0
        added = 0

        for j in range(3):
            if i + j >= len(rows):
                break
            row_text = format_markdown_row(rows[i + j])
            row_tokens = count_tokens(row_text)
            if token_total + row_tokens <= MAX_TOKENS:
                group.append(rows[i + j])
                token_total += row_tokens
                added += 1
            else:
                break

        if group:
            chunks.append(group)

        if added == 0:
            row = rows[i]
            temp_cells = []
            current_cells = []
            for cell in row:
                current_cells.append(cell)
                row_text = format_markdown_row(current_cells)
                if count_tokens(row_text) > MAX_TOKENS:
                    current_cells.pop()
                    if current_cells:
                        temp_cells.append(current_cells)
                    if count_tokens(cell) > MAX_TOKENS:
                        for split_cell in split_large_cell(cell):
                            temp_cells.append([split_cell])
                    current_cells = []
            if current_cells:
                temp_cells.append(current_cells)
            for small_row in temp_cells:
                chunks.append([small_row])
            i += 1
        else:
            i += added

    return chunks

def convert_chunks_to_markdown(chunks):
    markdown_list = []
    for chunk in chunks:
        max_cols = max(len(r) for r in chunk)
        md = []
        for row in chunk:
            row = row + [""] * (max_cols - len(row))
            md.append(format_markdown_row(row))
        markdown_list.append("\n".join(md))
    return markdown_list

def drop_collection(uri, db_name, collection_name):
    client = MilvusClient(
        uri=uri,
        token="root:Milvus",
        db_name=db_name
    )
    client.drop_collection(collection_name)

def create_tables_collection(uri, db_name, collection_name):
    client = MilvusClient(
        uri=uri,
        token="root:Milvus",
        db_name=db_name
    )
    schema = MilvusClient.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="table_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="document", datatype=DataType.VARCHAR, max_length=5096)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="type", 
        index_type="",
        index_name="type_index",
    )
    index_params.add_index(
        field_name="dense", 
        index_type="AUTOINDEX",
        index_name="dense_index",
        metric_type="COSINE"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def insert_collection(uri, db_name, collection_name, data):
    client = MilvusClient(
        uri=uri,
        token="root:Milvus",
        db_name=db_name
    )
    res = client.insert(
        collection_name=collection_name,
        data=data
    )
    return res

def search_collection(uri, db_name, collection_name, query_vector, filter_type, top_k):
    client = MilvusClient(
        uri=uri,
        token="root:Milvus",
        db_name=db_name
    )
    res = client.search(
        collection_name=collection_name,
        anns_field='dense',
        data=query_vector,
        limit=top_k,
        filter=f'type == "{filter_type}"',
        output_fields=["table_id", "type", "document"]
    )
    return res

def embedding_by_api(input):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": input
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    embedding_list =  [item['embedding'] for item in response.json()['data']]
    return embedding_list

def embedding_by_local(input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').to(device)
    model.eval()
    encoded_input = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:,0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()

def pad_image_to_height(img, target_height, fill_color=(255, 255, 255)):
    w, h = img.size
    if h >= target_height:
        return img
    pad_top = (target_height - h) // 2
    new_img = Image.new(img.mode, (w, target_height), fill_color)
    new_img.paste(img, (0, pad_top))
    return new_img

def proc_bbox(bbox,image_x_size):
    bbox[0] = 0
    bbox[1] -= 5
    bbox[2] = image_x_size
    bbox[3] += 5
    return bbox

def crop_images_from_pdfreader(middle_data: list, reader: PdfReader, page_item_idxs: dict):
    batch_writer = PdfWriter()
    bboxs = []
    table_images = []
    for page_idx, item_idxs in page_item_idxs.items():
        page_bboxs = []
        for item_idx in item_idxs:
            page_bboxs.append(middle_data[page_idx]['para_blocks'][item_idx]['bbox'])
        bboxs.append(page_bboxs)
        batch_writer.add_page(reader.pages[page_idx])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        batch_writer.write(tmp_pdf)
        tmp_pdf_path = tmp_pdf.name

    images = convert_from_path(tmp_pdf_path, dpi=72)

    for i, image in enumerate(images):
        for bbox in bboxs[i]:
            bbox = proc_bbox(bbox,image.size[0])
            table_image = image.crop(bbox)
            table_images.append(table_image)

    return table_images

def merge_images_horizonally(image1,image2):
    max_height = max(image1.height, image2.height)

    if image1.height < max_height:
        image1 = pad_image_to_height(image1, max_height)
    if image2.height < max_height:
        image2 = pad_image_to_height(image2, max_height)

    total_width = image1.width + image2.width
    merged_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (image1.width, 0))

    return merged_image

def contains_any(text, substrings):
    if isinstance(substrings, str):
        substrings = [substrings]
    return any(sub in text for sub in substrings)

def process_tables(content_list_path,middle_json_path,pdf_path,key_word=None):
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list_data = json.load(f)
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle_data = json.load(f)
    pdf_path = os.path.abspath(pdf_path)
    tables_list = []
    pending_table = None
    pending = False
    text_caption = None
    for item in content_list_data:
        if item["type"] == "table":
            if pending_table:
                
                
                if table_caption:
                    if pending_table:
                        tables_list.append(pending_table)
                        pending_table = None
                    if contains_any(str(table_caption),key_word):
                        item['table_id'] = str(uuid.uuid4())
                        item['pdf_path'] = pdf_path
                        pending_table = [item]
                else:  
                    if pending_table:
                        item['table_id'] = str(uuid.uuid4())
                        item['pdf_path'] = pdf_path
                        pending_table.append(item)
                    elif text_flag is True:
                        item['table_id'] = str(uuid.uuid4())
                        item['pdf_path'] = pdf_path
                        item['table_caption'] = [text_caption]
                        pending_table = [item]
                        text_flag = False
            else:
                
        elif item["type"] == "text":
            if pending:
                item_text = item['text']
                if contains_any(item_text,key_word):
                    pending = True
                    text_caption = item['text']
                tables_list.append(pending_table)
                pending_table = None
            else:
                item_text = item['text']
                if contains_any(item_text,key_word):
                    pending = True
                    text_caption = item['text']
            
    if pending_table:
        tables_list.append(pending_table)
   
    return tables_list

def process_ocr_data(json_path, pdf_path, key_word):
    tables_list = process_tables(json_path,pdf_path,key_word=key_word)
    for tables in tables_list:
        create_nodes(NEO4J_URI, NEO4J_AUTH,table_data=tables)
        for i, table in enumerate(tables):
            table_id = table['table_id']
            if i == 0:
                table_caption = table['table_caption']
                caption_embedding = embedding_by_api(table_caption)[0]
                caption_data = {
                    'table_id':table_id,
                    'type': 'caption',
                    'document':str(table_caption),
                    'dense':caption_embedding
                }
                insert_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',caption_data)
            if i < len(tables) - 1:
                create_relationship(NEO4J_URI, NEO4J_AUTH, table_id, tables[i+1]['table_id'])
            table_body = table['table_body']
            rows = html_to_rows(table_body)
            chunked_rows = chunk_rows(rows)
            markdown_chunks = convert_chunks_to_markdown(chunked_rows)
            embeddings = embedding_by_api(markdown_chunks)
            data = []
            for idx, chunk in enumerate(markdown_chunks):
                item = {
                    'table_id':table_id,
                    'type': 'content',
                    'document':chunk,
                    'dense':embeddings[idx]
                }
                data.append(item)
            insert_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',data)

def search_by_text(input, doc_type='caption', top_k=1):
    embeddings = embedding_by_api(input)
    print("完成所有输入的嵌入.")
    print('对所有输入，检索向量数据库...')
    doc_res = search_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',embeddings,doc_type,top_k)
    results = []
    for i, item in enumerate(doc_res):
        print("="*20)
        print('输入：\n',input[i])
        print('向量数据库检索结果：\n',item)
        print('检索知识图谱...')
        nodes = search_nodes(NEO4J_URI,NEO4J_AUTH,item[0]['entity']['table_id'])[0]
        page_idx_list = []
        for j, node in enumerate(nodes):
            if j == 0:
                pdf_path = node["t"]['pdf_path']
            page_idx_list.append(node["t"]['page_idx'])
        print('知识图谱检索结果：')
        print('pdf文件路径：\n',pdf_path)
        print('pdf页码：\n',page_idx_list)
        page_images = convert_from_path(pdf_path,dpi=300)
        results.append([])
        for page_idx in page_idx_list:
            results[i].append(page_images[page_idx])
            
    return results

if __name__ == "__main__":
    # drop_collection(MILVUS_URI,MILVUS_DB_NAME,'tables')
    # create_tables_collection(MILVUS_URI,MILVUS_DB_NAME,'tables')
    # clear_graph(NEO4J_URI, NEO4J_AUTH)
    # process_ocr_data('./output/江苏华鹏投标技术文件_content_list.json','./pdfs/江苏华鹏投标技术文件.pdf',key_word=['设备外部条件一览表','工程概况一览表'])
    # input = '工程概况一览表'
    # results = search_by_text(input,'caption',1)
    # for result in results:
    #     for image in result:
    #         plt.imshow(image)
    #         plt.axis('off')
    #         plt.show()
    # tables_list = process_tables_for_user_input('./output/投标文件/auto/投标文件_content_list.json','./pdfs/投标文件.pdf')
    # for tables in tables_list:
    #     print(tables)

    

    with open('./output/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版_middle.json', 'r', encoding='utf-8') as f:
        middle_data = json.load(f)

    reader = PdfReader('./pdfs/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版.pdf')

    page_item_idxs = {
        2:[3],
        3:[2]
    }

    table_images = crop_images_from_pdfreader(middle_data, reader, page_item_idxs)

    for image in table_images:
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    merge_image = merge_images_horizonally(table_images[0], table_images[1])
    merge_image.save('merged.png')