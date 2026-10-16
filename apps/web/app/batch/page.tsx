import { AppHeader } from "@/components/app-header";
import { BatchView } from "@/components/batch-view";

export default async function BatchPage({
  searchParams,
}: {
  searchParams: Promise<{ ids?: string; failed?: string }>;
}) {
  const { ids, failed } = await searchParams;
  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page">
        <BatchView idsParam={ids ?? ""} failedParam={failed ?? ""} />
      </main>
    </div>
  );
}
