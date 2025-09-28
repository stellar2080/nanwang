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

NEO4J_URI = "bolt://localhost:18802"
NEO4J_AUTH = ("neo4j", "12345678")
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = 'default'
MAX_TOKENS = 256
SILICONFLOW_API_KEY = 'sk-lfgwvzyqxmwxtomwdqqjbdlvibfdrravglobjhiuvnqnfwyx'
TONGYI_API_KEY = 'sk-9536a97947b641ad9e287da238ba3abb'
PDF_DPI = 300

template_for_tablename_extraction = """
你是一个表格分析专家。你将看到一张图片和该图片中某一个表格的OCR识别结果。
你的任务是结合图像内容和OCR识别结果，识别该表可能性最大的表名。
【OCR识别结果】
{}
【任务要求】
1.表名通常位于表格上方，可能以以下形式出现：
- 表格编号与名称组合形式，如“表x.x 统计表”或“表x-x 信息表”
- 带章节编号的标题，如“1.1 xxx表”、“3.2 数据统计表”
- 纯文字标题，如“汇总表”、“概况一览表”等
2.若存在多个候选表名，请选择位置最贴近表格且结构最符合上述形式的一项
3.注意排除页眉、页脚、段落正文、图注等干扰信息。
【输出要求】
请严格按以下格式输出识别到的可能性最大的表名，绝对不允许留空：
{{
    "name": "识别到的表名"
}}
"""

template_for_header_judge = '''
你是一个表格分析专家，你将看到两个以HTML格式提供的表格内容。
你的任务是严格逐步执行【分析步骤】，禁止跳步。
【表格内容】
第一个表格：
{}
第二个表格：
{}
【分析步骤】
1.判断两个表格是否含有表头
以下是表头行和非表头行的特征：
   - 表头行：用于标识表格列含义的标题行，通常包含以下类型的词语：序号、名称、项目、参数、数量、单位、条件、备注、类型、值、类别  
   - 非表头行：不是列标题，而是数据行、说明行或数值行，例如：
       - "1.1 | 电动机 | 2台"
       - "2.3.1 | 实操培训时长（天）| 10 | 10"
       - "环境温度 | 25℃"   
       - "2024年7月 | XX公司"
a.如果都有表头 → 进入第 2 步
b.如果至少一个表格没有表头，输出 {{"judge": 1}} 并立即结束
2.对比两个表格的表头内容与结构是否完全一致
a.如果一致，输出 {{"judge": 1}} 并立即结束
b.如果不一致，输出 {{"judge": 0}} 并立即结束
【输出要求】
请逐步输出分析过程，中间分析过程请用自然语言描述，最终判断结果必须单独按以下格式输出：
{{
    "judge": 1
}}
或
{{
    "judge": 0
}}
'''

template_for_table_serial_and_continuity_check = '''
你是一个表格分析专家，你将看到两个以HTML格式提供的表格内容。
你的任务是严格逐步执行【分析步骤】，禁止跳步。
【表格内容】
第一个表格：
{}
第二个表格：
{}
【分析步骤】
1.序号列连贯性检查
a.如果两个表格都有明确的序号列，执行以下步骤：
    i. 提取第一个表格序号列的最后一个序号（记为 left_last）和第二个表格序号列的第一个序号（记为 right_first）。
    ii. 将 left_last 和 right_first 按照以下的“版本号比较”规则进行解析与比较：
    - 将每个序号以点（.）分割成多个部分，每个部分转换为整数。
    - 从前到后逐段比较：
    - 若某一段左 < 右 → 整体左 < 右 → 视为连贯
    - 若某一段左 > 右 → 不连贯
    - 若相等，继续比较下一段
    - 如果所有已有的段都相等，则较短的序号视为“更小”（例如 2.3 < 2.3.1）
    - 示例：
    - 2.3 vs 2.4 → 2==2, 3<4 → 连贯
    - 2.6 vs 2.6.1 → 前两段相等，左更短 → 左 < 右 → 连贯
    - 2.3.1 vs 2.4 → 2==2, 3<4 → 连贯
    - 3 vs 2.99 → 3>2 → 不连贯
    iii. 如果按上述规则比较得到 left_last < right_first → 输出 {{"check": 1}} 并立即结束
    iv. 否则（即 left_last >= right_first）→ 输出 {{"check": 0}} 并立即结束
b.如果至少有一个表格没有明确的序号列 → 进入第 2 步
2. 内容连续性判断
你需要按以下规则判断两个表格的内容是否连续：
- 内容连续是指两个表格在阅读顺序上属于同一部分，后一个表格延续、补充或并列展开前一个表格的内容。
- 可以通过以下线索判断为连续：
   1) 后一个表格继续了前一个表格的小节、主题，或在同一章节下展开另一个并列小节。
   2) 内容明显属于同一大类或同一小节的不同方面（如同一章下不同系统、不同工艺要求）。
判断规则：
a. 如果表格中内容连续 → 输出 {{"check": 1}}
b. 如果表格中内容不连续 → 输出 {{"check": 0}}
【输出要求】
请逐步输出分析过程，中间分析过程请用自然语言描述，最终判断结果必须单独按以下格式输出：
{{
    "check": 1
}}
或
{{
    "check": 0
}}
'''


def image_to_base64_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_base64_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def pdf_to_base64_images_by_batch(reader: PdfReader, start_page=0, end_page=None, batch_size=30):
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

        images = convert_from_path(tmp_pdf_path,dpi=PDF_DPI)
        for img in images:
            base64_images.append(image_to_base64_from_pil(img))

        os.remove(tmp_pdf_path)

    return base64_images

def vl_chat_from_path(image_path, prompt):
    client = OpenAI(
        api_key=TONGYI_API_KEY,
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
        temperature=0.1,
    )
    return response.choices[0].message.content

def vl_chat(base64_image, prompt):
    client = OpenAI(
        api_key=TONGYI_API_KEY,
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
        temperature=0.1,
    )
    return response.choices[0].message.content

def chat(prompt):
    client = OpenAI(
        api_key=TONGYI_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[
            {
                "role": "user","content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

def parse_json(content: str) -> dict:
    try:
        start = content.rfind("{")
        if start == -1:
            raise ValueError("No '{' found in content")

        end = content.rfind("}")
        if end == -1:
            raise ValueError("No '}' found after last '{'")

        json_string = content[start:end+1]

        return json.loads(json_string)

    except Exception as e:
        print(f"parse json error: {e}\ncontent: {content}")
        return None

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
                pdf_path: table.pdf_path,
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

def search_chains(url: str, auth: tuple, table_id: str):
    with GraphDatabase.driver(url, auth=auth) as driver:
        driver.verify_connectivity()
        result = driver.execute_query("""
            MATCH (start:Table {table_id: $table_id})
            WITH start.serial_idx AS serial_idx

            MATCH (n:Table {serial_idx: serial_idx})

            MATCH (n)-[:NEXT*0..]-(m:Table {serial_idx: serial_idx})
            WITH n, collect(elementId(m)) AS ids

            WITH n, apoc.coll.min(ids) AS componentId

            ORDER BY componentId, size(n.table_caption) > 0 DESC, n.page_idx ASC
            RETURN componentId, collect(n) AS chain
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

def query_collection(uri, db_name, collection_name, filter_type):
    client = MilvusClient(
        uri=uri,
        token="root:Milvus",
        db_name=db_name
    )
    res = client.query(
        collection_name=collection_name,
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

def proc_bbox(bbox: list,image_x_size,mode: int):
    if mode == 0:
        bbox[0] = 0
        bbox[1] = 0
        bbox[2] = image_x_size
        bbox[3] += 60
    elif mode == 1:
        bbox[0] = 0
        bbox[1] -= 60
        bbox[2] = image_x_size
        bbox[3] += 60
    return bbox

def crop_images_from_pdfreader(reader: PdfReader, bboxs: list, mode: int = 1):
    batch_writer = PdfWriter()
    table_images = []
    for item in bboxs:
        page_idx = item[0]
        batch_writer.add_page(reader.pages[page_idx])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        batch_writer.write(tmp_pdf)
        tmp_pdf_path = tmp_pdf.name

    images = convert_from_path(tmp_pdf_path, dpi=PDF_DPI)

    scale = PDF_DPI / 72
    for i, image in enumerate(images):
        bbox = bboxs[i][1]
        bbox = [int(x * scale) for x in bbox]
        bbox = proc_bbox(bbox,image.size[0],mode)
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

def extract_table_name(item, bboxs, reader, pdf_path):
    """裁剪表格图像并调用模型识别表名，返回更新后的 item"""
    table_image = crop_images_from_pdfreader(reader, bboxs, mode=0)[0]
    # plt.imshow(table_image)
    # plt.axis('off')
    # plt.show()
    base64_image = image_to_base64_from_pil(table_image)
    content = vl_chat(base64_image,prompt=template_for_tablename_extraction.format(item['table_body']))
    print('tablename_extraction\n大模型回答：\n',content)
    json_res = parse_json(content)
    print('识别表名为:',json_res['name'])
    item['table_id'] = str(uuid.uuid4())
    item['pdf_path'] = pdf_path
    item['table_caption'] = json_res['name']
    item['bbox'] = bboxs[-1][1]
    return item

def truncate_by_tr(s):
    target = "</tr>"
    count = 0
    start = 0

    while count < 4:
        pos = s.find(target, start)
        if pos == -1:
            return s
        count += 1
        if count == 4:
            end_pos = pos + len(target)
            return s[:end_pos]
        start = pos + len(target)

    return s

def truncate_by_tr_from_end(s):
    target = "<tr>"
    count = 0
    pos = len(s)

    while count < 4:
        found_pos = s.rfind(target, 0, pos)
        if found_pos == -1:
            return s
        count += 1
        if count == 4:
            return s[found_pos:]
        pos = found_pos

    return s

def process_tables(content_list_path,pdf_path):
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list_data = json.load(f)
    pdf_path = os.path.abspath(pdf_path)
    reader = PdfReader(pdf_path)
    tables_list = []
    pending_table = None
    last_table_idx = -1
    for idx, item in enumerate(content_list_data):
        print('='*30)
        print('当前item信息:')
        print(item)
        page_idx = item['page_idx']
        bbox = item['bbox']
        bboxs = [(page_idx, bbox)]
            
        if item["type"] == "table" and 'table_body' in item:
            if pending_table: 
                print('pending_table不为空.')
                table_body = item['table_body']
                last_table_body = content_list_data[last_table_idx]['table_body']
                prompt = template_for_header_judge.format(truncate_by_tr(last_table_body),truncate_by_tr(table_body))
                print('template_for_header_judge:\n',prompt)
                content = chat(prompt=prompt)
                print('大模型回答：\n',content)
                judge_res = parse_json(content)
                if judge_res['judge'] == 1:
                    prompt = template_for_table_serial_and_continuity_check.format(truncate_by_tr_from_end(last_table_body),truncate_by_tr(table_body))
                    print('template_for_table_serial_and_continuity_check:\n',prompt)
                    content = chat(prompt=prompt)
                    print('大模型回答：\n',content)
                    check_res = parse_json(content)

                    if check_res['check'] == 1:
                        item['table_id'] = str(uuid.uuid4())
                        item['pdf_path'] = pdf_path
                        item['table_caption'] = ""
                        item['bbox'] = bbox
                        pending_table.append(item)       
                    elif check_res['check'] == 0:
                        tables_list.append(pending_table)
                        pending_table = None
                        item = extract_table_name(item,bboxs,reader,pdf_path)
                        pending_table = [item]

                elif judge_res['judge'] == 0:
                    tables_list.append(pending_table)
                    pending_table = None
                    item = extract_table_name(item,bboxs,reader,pdf_path)
                    pending_table = [item]

            else:
                print('pending_table为空.')
                item = extract_table_name(item,bboxs,reader,pdf_path)
                pending_table = [item]

            last_table_idx = idx
                
        # print('当前pending_table信息:')
        # print(pending_table)

    if pending_table:
        tables_list.append(pending_table)
   
    return tables_list

def process_ocr_data(content_list_path, pdf_path, key_word):
    tables_list = process_tables(content_list_path,pdf_path)
    for tables in tables_list:
        for i, table in enumerate(tables):
            table_id = table['table_id']
            table_caption = table['table_caption']
            if i == 0 and key_word.replace(' ','') in table_caption.replace(' ',''): # 检查是否是一串表中的第一个表，如工程概况一览表，若是，则进行嵌入
                caption_embedding = embedding_by_api(table_caption)[0]
                caption_data = {
                    'table_id':table_id,
                    'type': 'caption',
                    'document':str(table_caption),
                    'dense':caption_embedding
                }
                insert_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',caption_data)
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
            else:
                caption_data = {
                    'table_id':table_id,
                    'type': 'caption',
                    'document':str(table_caption)
                }
                insert_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',caption_data)
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
    result = []
    for i, item in enumerate(doc_res):
        print("="*20)
        print('输入：\n',input[i])
        print('向量数据库检索结果：\n',item)
        print('检索知识图谱...')
        chains = search_chains(NEO4J_URI,NEO4J_AUTH,item[0]['entity']['table_id'])[0]
        bboxs = []
        pdf_path = chains[0]['chain'][0]['pdf_path']
        for chain in chains:    
            for node in chain['chain']:
                bboxs.append((node['page_idx'],node['bbox']))
        print('知识图谱检索结果：')
        print('pdf文件路径：\n',pdf_path)
        print('bbox信息：\n',bboxs)
        pdfreader = PdfReader(pdf_path)
        images = crop_images_from_pdfreader(pdfreader, bboxs, mode=1)
        result.append(images)
    
    return result

if __name__ == "__main__":
    drop_collection(MILVUS_URI,MILVUS_DB_NAME,'tables')
    create_tables_collection(MILVUS_URI,MILVUS_DB_NAME,'tables')
    clear_graph(NEO4J_URI, NEO4J_AUTH)
    process_ocr_data(
        './output/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版_content_list.json',
        './pdfs/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版.pdf',
        key_word='工程概况一览表'
    )

    # input = ['<html><body><table><tr><td>序号</td><td>名称</td><td>内容</td></tr><tr><td>1</td><td>工程名称</td><td>500千伏楚庭站扩建第三台主变工程</td></tr><tr><td>2</td><td>工程建设单位</td><td>广东电网有限责任公司广州供电局</td></tr><tr><td>3</td><td>工程地址</td><td>广州市番禺区</td></tr><tr><td></td><td>是否为扩建工程（是/否）</td><td>是</td></tr><tr><td></td><td>工程规模</td><td>1×1000MVA</td></tr><tr><td>6</td><td>运输条件</td><td>陆运</td></tr><tr><td>，</td><td>电气主接线</td><td>']
    # result = search_by_text(input,'content')
    # for images in result:
    #     for image in images:
    #         plt.imshow(image)
    #         plt.axis('off')
    #         plt.show()