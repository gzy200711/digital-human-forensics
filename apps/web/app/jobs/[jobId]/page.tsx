import { AppHeader } from "@/components/app-header";
import { ReportView } from "@/components/report-view";

export default async function JobPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = await params;
  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page">
        <ReportView jobId={jobId} />
      </main>
    </div>
  );
}

