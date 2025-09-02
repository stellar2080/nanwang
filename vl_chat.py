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

template_for_table_firstrow_classification = '''
你是一个表格分析专家，你将看到一张包含两个表格的图像。
你的任务是根据【判断规则】分别判断两个表格的首行是否为列名行
【判断规则】
1.如果表格有标题，不要把标题误认为是首行内容
2.只能根据首行原始文本进行判断，禁止解释、拆解、补充或推理首行内容含义
3.列名行：用于标识表格列含义的标题行，通常包含以下类型的词语：
    - 示例：序号、名称、项目、参数、数量、单位、条件、备注、类型、值、类别
    - 示例完整行：
      - "序号 | 名称 | 数量"
      - "参数 | 单位 | 正常使用条件 | 特殊使用条件"
      - "项目 | 类型 | 备注"
4.非列名行：不是列标题，而是数据行、说明行或数值行，通常不具备抽象的列标题性质。
    - 示例：序号值、具体值、数字、时间、公司名、设备型号，而不是抽象的列概念
    - 示例完整行：
      - "1.1 | 电动机 | 2台"
      - "2.3.1 |实操培训时长（天）|10|10"
      - "环境温度 | 25℃"
      - "2024年7月 | XX公司"
5.强制规则：
    - 如果首行第一个单元格是具体的编号值（例如纯数字、罗马数字、带小数点的序号，如“1”“1.1”“1.2.3.4”“一”），则必定为非列名行。
【输出要求】
请逐步输出判断过程，中间判断过程请用自然语言描述，最终判断结果必须单独按以下格式输出：
- 如果两个表格的首行都是列名行:
{
    "both_are_headers": true
}
- 如果至少有一个表格首行不是列名行，输出:
{
    "both_are_headers": false
}
'''

template_for_table_firstrow_comparison = '''
你是一个表格分析专家，你将看到一张包含两个表格的图像。
你的任务是严格基于【判断规则】对比两个表格的首行表头是否完全相同。
【判断规则】
1.你只需判断两个表格首行表头是否完全一致，不需要考虑其他行。
2.必须基于图像内容进行直接对比，禁止解释、推理或扩展含义。
3.你必须特别注意表格是否存在多级表头
【输出要求】
请逐步输出判断过程，中间判断过程请用自然语言描述，最终判断结果必须单独按以下格式输出：
- 如果完全一致，输出：
{
    "firstrow_are_equal": true
}
- 如果不一致，输出：
{
    "firstrow_are_equal": false
}
'''

template_for_table_serial_and_continuity_check = '''
你是一个表格分析专家，你将看到一张含有两个表格的图片。
你的任务是严格逐步执行【判断步骤】，禁止跳步。
【判断步骤】
1.序号连贯性检查
a.如果两个表格都有序号，执行以下步骤：
    i. 提取左表格内最后一个序号（记为 left_last）和右表格内第一个序号（记为 right_first）。
    - 注意，不要将标题或表名的编号误认为是表格内的序号
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
    iii. 如果按上述规则比较得到 left_last < right_first → 输出 {"type": 1} 并立即结束
    iv. 否则（即 left_last >= right_first）→ 输出 {"type": 0} 并立即结束
b.如果至少有一个表格没有序号 → 进入第 2 步
2.内容衔接性判断
注意：内容连续仅指在阅读顺序上表格行内容明显延续，不允许仅凭“主题相关/设备相同”判断。
a.如果表格中内容连续 → 输出 {"type": 1}
b.如果表格中内容不连续 → 输出 {"type": 0}
【输出要求】
请逐步输出判断过程，中间判断过程请用自然语言描述，最终判断结果必须单独按以下格式输出：
{
    "type": 1
}
或
{
    "type": 0
}
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
        temperature=0.4,
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
        temperature=0.4,
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
                bbox: table.bbox
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

def get_bboxs(middle_data: list, idx_data: list):
    bboxs = []
    for item in idx_data:
        page_idx, item_idx = item
        bboxs.append((page_idx,middle_data[page_idx]['para_blocks'][item_idx]['bbox']))
    return bboxs

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
    table_images = crop_images_from_pdfreader(reader, bboxs[-1:], mode=0)
    table_image = table_images[-1]
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

def process_tables(content_list_path,middle_json_path,pdf_path):
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list_data = json.load(f)
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle_data = json.load(f)
    pdf_path = os.path.abspath(pdf_path)
    reader = PdfReader(pdf_path)
    tables_list = []
    pending_table = None

    idx_data = []
    page_idx_now = 0
    item_idx = 0
    for item in content_list_data:
        print('='*30)
        # print('当前item信息:')
        # print(item)
        page_idx = item['page_idx']
        if page_idx_now != page_idx:
            page_idx_now = page_idx
            item_idx = 0 
        if item["type"] == "table": 
            idx_data.append((page_idx,item_idx))
            # print('当前idx_data信息:')
            # print(idx_data)
            if pending_table: 
                print('pending_table不为空.')
                print('截取pdf中表格.')
                bboxs = get_bboxs(middle_data, idx_data[-2:])
                table_images = crop_images_from_pdfreader(reader, bboxs, mode=1)
                print('合并表格.')
                merge_image = merge_images_horizonally(table_images[0], table_images[1])
                plt.imshow(merge_image)
                plt.axis('off')
                plt.show()
                base64_image = image_to_base64_from_pil(merge_image)
                content = vl_chat(base64_image,prompt=template_for_table_firstrow_classification)
                print('table_firstrow_classification\n大模型回答：\n',content)
                classify_res = parse_json(content)
                if classify_res['both_are_headers']:
                    content = vl_chat(base64_image,prompt=template_for_table_firstrow_comparison)
                    print('table_firstrow_comparison\n大模型回答：\n',content)
                    firstrow_comparison_res = parse_json(content)

                    if firstrow_comparison_res['firstrow_are_equal']:
                        content = vl_chat(base64_image,prompt=template_for_table_serial_and_continuity_check)
                        print('table_serial_and_continuity_check\n大模型回答：\n',content)
                        check_res = parse_json(content)
                        if check_res['type'] == 1:
                            item['table_id'] = str(uuid.uuid4())
                            item['pdf_path'] = pdf_path
                            item['table_caption'] = ""
                            item['bbox'] = bboxs[-1][1]
                            pending_table.append(item)
                        elif check_res['type'] == 0:
                            tables_list.append(pending_table)
                            pending_table = None
                            
                            item = extract_table_name(item,bboxs,reader,pdf_path)
                            pending_table = [item]
                            
                    else:
                        tables_list.append(pending_table)
                        pending_table = None
                        
                        item = extract_table_name(item,bboxs,reader,pdf_path)
                        pending_table = [item]

                else:
                    content = vl_chat(base64_image,prompt=template_for_table_serial_and_continuity_check)
                    print('table_serial_and_continuity_check\n大模型回答：\n',content)
                    check_res = parse_json(content)
                    if check_res['type'] == 1:
                        item['table_id'] = str(uuid.uuid4())
                        item['pdf_path'] = pdf_path
                        item['table_caption'] = ""
                        item['bbox'] = bboxs[-1][1]
                        pending_table.append(item)
                    elif check_res['type'] == 0:
                        tables_list.append(pending_table)
                        pending_table = None
                        
                        item = extract_table_name(item,bboxs,reader,pdf_path)
                        pending_table = [item]


            else:
                print('pending_table为空.')
                bboxs = get_bboxs(middle_data, idx_data[-1:])

                item = extract_table_name(item,bboxs,reader,pdf_path)
                pending_table = [item]
                
        # print('当前pending_table信息:')
        # print(pending_table)
        item_idx += 1

    if pending_table:
        tables_list.append(pending_table)
   
    return tables_list

def process_ocr_data(json_path, middle_json_path, pdf_path, key_word):
    tables_list = process_tables(json_path,middle_json_path,pdf_path)
    for tables in tables_list:
        create_nodes(NEO4J_URI, NEO4J_AUTH,table_data=tables)
        embedding_flag = False
        for i, table in enumerate(tables):
            table_id = table['table_id']
            table_caption = table['table_caption']
            if i < len(tables) - 1:
                create_relationship(NEO4J_URI, NEO4J_AUTH, table_id, tables[i+1]['table_id'])
            if i == 0 and key_word.replace(' ','') in table_caption.replace(' ',''): # 检查是否是一串表中的第一个表，如工程概况一览表，若是，则进行嵌入
                embedding_flag = True
            if embedding_flag:
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
        nodes = search_nodes(NEO4J_URI,NEO4J_AUTH,item[0]['entity']['table_id'])[0]
        bboxs = []
        for j, node in enumerate(nodes):
            if j == 0:
                pdf_path = node["t"]['pdf_path']
            bboxs.append((node["t"]['page_idx'],node["t"]['bbox']))
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
        './output/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版_middle.json',
        './pdfs/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版.pdf',
        key_word='工程概况一览表'
    )
    # input = ['工程概况一览表']
    # result = search_by_text(input,'caption')
    # for images in result:
    #     for image in images:
    #         plt.imshow(image)
    #         plt.axis('off')
    #         plt.show()


    # content_list_path = './output/9、500kV组合电器-河南平芝1_投标技术文件_content_list.json'
    # middle_json_path = './output/9、500kV组合电器-河南平芝1_投标技术文件_middle.json'
    # pdf_path = './pdfs/9、500kV组合电器-河南平芝1_投标技术文件.pdf'
    # image_name = '河南平芝_'
    # content_list_path = './output/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版_content_list.json'
    # middle_json_path = './output/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版_middle.json'
    # pdf_path = './pdfs/12、500千伏楚庭站扩建第三台主变工程550kVGIS 技术确认书--盖章版.pdf'
    # image_name = '楚庭站_'
    # with open(content_list_path, 'r', encoding='utf-8') as f:
    #     content_list_data = json.load(f)
    # with open(middle_json_path, 'r', encoding='utf-8') as f:
    #     middle_data = json.load(f)
    # pdf_path = os.path.abspath(pdf_path)
    # reader = PdfReader(pdf_path)
    # idx_data = []
    # page_idx_now = 0
    # item_idx = 0
    # for item in content_list_data:
    #     print('='*30)
    #     print('当前item信息:')
    #     print(item)
    #     page_idx = item['page_idx']
    #     if page_idx_now != page_idx:
    #         page_idx_now = page_idx
    #         item_idx = 0 
    #     if item["type"] == "table": 
    #         idx_data.append((page_idx,item_idx))
    #         print('当前idx_data信息:')
    #         print(idx_data)
    #     item_idx += 1
    # table_images = crop_images_from_pdfreader(middle_data, reader, idx_data, mode=1)
    # if not os.path.exists('./images'):
    #     os.mkdir('./images')
    # for idx, image in enumerate(table_images):
    #     if idx > 10:
    #         break
    #     image.save(os.path.join('images',image_name+str(idx)+'.jpg'))
    # img = Image.open('./images/楚庭站_6.jpg')
    # img1 = Image.open('./images/河南平芝_10.jpg')
    # merge_img = merge_images_horizonally(img,img1)
    # merge_img.save('merge4.jpg')

    # res = query_collection(MILVUS_URI,MILVUS_DB_NAME,'tables','caption')
    # caption_list = []
    # for item in res:
    #     caption_list.append(item['document'])
    # print(caption_list)

    # middle_json_path = './output/9、500kV组合电器-河南平芝1_投标技术文件_middle.json'
    # content_list_path = './output/9、500kV组合电器-河南平芝1_投标技术文件_content_list.json'
    # pdf_path = './pdfs/9、500kV组合电器-河南平芝1_投标技术文件.pdf'
    # with open(content_list_path, 'r', encoding='utf-8') as f:
    #     content_list_data = json.load(f)
    # with open(middle_json_path, 'r', encoding='utf-8') as f:
    #     middle_data = json.load(f)
    
    # idx_data = []
    # reader = PdfReader(pdf_path)
    # for item in content_list_data:
    #     print('='*30)
    #     print('当前item信息:')
    #     print(item)
    #     page_idx = item['page_idx']
    #     if page_idx_now != page_idx:
    #         page_idx_now = page_idx
    #         item_idx = 0 
    #     if item["type"] == "table": 
    #         idx_data.append((page_idx,item_idx))
    #         print('当前idx_data信息:')
    #         print(idx_data)

    #         bboxs = get_bboxs(middle_data, idx_data[-1:])
    #         table_images = crop_images_from_pdfreader(reader, bboxs[-1:],mode=0)
    #         table_image = table_images[-1]
    #         plt.imshow(table_image)
    #         plt.axis('off')
    #         plt.show()
    #         base64_image = image_to_base64_from_pil(table_image)
    #         content = vl_chat(base64_image,prompt=template_for_tablename_extraction.format(item['table_body'],caption_list))
    #         print('大模型回答：\n',content)
    #         json_res = parse_json(content)
    #         if json_res['name']:
    #             table_name = json_res['name']
    #             print('识别表名为:',table_name)
    #             table_body = item['table_body']
    #             rows = html_to_rows(table_body)
    #             chunked_rows = chunk_rows(rows)
    #             markdown_chunks = convert_chunks_to_markdown(chunked_rows)
    #             markdown_chunks.insert(0,table_name)
    #             embeddings = embedding_by_api(markdown_chunks)
    #             res = search_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',embeddings[0],'caption',top_k=50000)
    #             print(res)
    #     item_idx += 1

    # embeddings = embedding_by_api(['供货范围及设备技术规格一览表'])
    # res = search_collection(MILVUS_URI,MILVUS_DB_NAME,'tables',embeddings,'caption',top_k=50000)
    # print(res)