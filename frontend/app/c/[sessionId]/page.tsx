import ChatPage from "@/components/chat/chat-page";

export default async function ChatSessionPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = await params;
  return <ChatPage routeSessionId={sessionId} />;
}
