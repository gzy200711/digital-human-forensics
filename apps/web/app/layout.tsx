import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "数字人鉴别平台",
  description: "证据优先的数字人 / 生成合成内容鉴别平台",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

