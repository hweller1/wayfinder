export interface Session {
  token: string;
  user: string;
}

export async function getUserSession(token: string): Promise<Session | null> {
  const res = await fetch(`/api/session/${token}`);
  return res.ok ? ((await res.json()) as Session) : null;
}
