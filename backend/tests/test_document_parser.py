from app.services.document_parser import parse_document


def test_text_document_is_split_into_searchable_chunks() -> None:
    """UTF-8 文本资料应保留全文，并切分为至少一个可检索片段。"""
    parsed = parse_document("活动规则.md", "报名时间为 8 月 1 日至 8 月 7 日。\n商品需要满足库存要求。".encode())

    assert "报名时间" in parsed.content
    assert parsed.chunks[0].page_number is None
    assert "库存要求" in parsed.chunks[0].content


def test_pdf_document_keeps_page_number() -> None:
    """PDF 资料片段必须保留页码，以便问答结果给出可复核引用。"""
    import fitz

    source = fitz.open()
    page = source.new_page()
    # PyMuPDF 的默认测试字体不含中文；真实中文 PDF 会携带自身的字体与文本映射。
    page.insert_text((72, 72), "Registration requires store verification.")
    raw_content = source.tobytes()
    source.close()

    parsed = parse_document("规则.pdf", raw_content)

    assert parsed.chunks[0].page_number == 1
    assert "Registration" in parsed.chunks[0].content


def test_docx_document_extracts_paragraphs() -> None:
    """DOCX 资料应从段落中提取可检索文本。"""
    from docx import Document
    from io import BytesIO

    document = Document()
    document.add_paragraph("优惠券仅限活动期间领取。")
    buffer = BytesIO()
    document.save(buffer)

    parsed = parse_document("规则.docx", buffer.getvalue())

    assert "优惠券" in parsed.content


def test_unsupported_document_type_is_rejected() -> None:
    """未实现解析器的文件不能被错误地纳入知识库。"""
    try:
        parse_document("活动规则.xlsx", b"not an excel file")
    except ValueError as error:
        assert "仅支持" in str(error)
    else:
        raise AssertionError("应拒绝不支持的文件格式")
