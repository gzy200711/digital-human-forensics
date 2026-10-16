import { NextRequest, NextResponse } from "next/server";

function firstHeaderValue(value: string | null): string {
  return (value ?? "").split(",", 1)[0]?.trim().toLowerCase() ?? "";
}

function hostnameWithoutPort(value: string): string {
  if (value.startsWith("[")) {
    const closing = value.indexOf("]");
    return closing >= 0 ? value.slice(1, closing) : value;
  }
  const lastColon = value.lastIndexOf(":");
  if (lastColon > -1 && value.indexOf(":") === lastColon) {
    return value.slice(0, lastColon);
  }
  return value;
}

function isPrivateIpv4(hostname: string): boolean {
  const parts = hostname.split(".");
  if (parts.length !== 4 || parts.some((part) => !/^\d{1,3}$/.test(part))) {
    return false;
  }
  const [a, b] = parts.map(Number);
  if ([a, b, ...parts.slice(2).map(Number)].some((value) => value > 255)) {
    return false;
  }
  return (
    a === 10 ||
    a === 127 ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168)
  );
}

function isPrivateHost(hostname: string): boolean {
  if (!hostname) return false;
  if (
    hostname === "localhost" ||
    hostname.endsWith(".localhost") ||
    hostname.endsWith(".local") ||
    hostname === "::1" ||
    (hostname.includes(":") &&
      (hostname.startsWith("fc") ||
        hostname.startsWith("fd") ||
        hostname.startsWith("fe80:")))
  ) {
    return true;
  }
  return isPrivateIpv4(hostname);
}

function timingSafeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

function unauthorized(): NextResponse {
  return new NextResponse("Authentication required.", {
    status: 401,
    headers: {
      "Cache-Control": "no-store",
      "WWW-Authenticate": 'Basic realm="DHFP", charset="UTF-8"',
    },
  });
}

export default function proxy(request: NextRequest): NextResponse {
  const forwardedHost = firstHeaderValue(request.headers.get("x-forwarded-host"));
  const directHost = firstHeaderValue(request.headers.get("host"));
  const hostname = hostnameWithoutPort(forwardedHost || directHost);
  const viaCloudflare =
    request.headers.has("cf-ray") || request.headers.has("cf-connecting-ip");

  if (!viaCloudflare && isPrivateHost(hostname)) {
    return NextResponse.next();
  }

  const expectedUser = process.env.PUBLIC_ACCESS_USER ?? "";
  const expectedPassword = process.env.PUBLIC_ACCESS_PASSWORD ?? "";
  if (!expectedUser || !expectedPassword) {
    return new NextResponse("Public access is not configured.", {
      status: 503,
      headers: { "Cache-Control": "no-store" },
    });
  }

  const authorization = request.headers.get("authorization") ?? "";
  if (!authorization.toLowerCase().startsWith("basic ")) {
    return unauthorized();
  }

  try {
    const decoded = atob(authorization.slice(6).trim());
    const separator = decoded.indexOf(":");
    if (separator < 0) return unauthorized();
    const suppliedUser = decoded.slice(0, separator);
    const suppliedPassword = decoded.slice(separator + 1);
    if (
      timingSafeEqual(suppliedUser, expectedUser) &&
      timingSafeEqual(suppliedPassword, expectedPassword)
    ) {
      return NextResponse.next();
    }
  } catch {
    return unauthorized();
  }

  return unauthorized();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
