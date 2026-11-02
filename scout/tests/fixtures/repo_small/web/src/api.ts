import { getUserSession } from "./session";

export async function whoami(token: string): Promise<string> {
  const session = await getUserSession(token);
  return session ? session.user : "anonymous";
}
