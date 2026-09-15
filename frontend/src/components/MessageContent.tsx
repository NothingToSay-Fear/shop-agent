import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * 使用标准 Markdown 渲染器解析 Agent 输出；GFM 负责表格等扩展语法，
 * 外层组件仅定义运营工作台的阅读样式，不执行原始 HTML。
 */
export function MessageContent({ content }: { content: string }) {
  return (
    <div className="message-rich-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="message-table-wrap">
              <table className="message-table">{children}</table>
            </div>
          ),
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer">{children}</a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
