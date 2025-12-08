import { Header } from './Header';
import { Footer } from './Footer';

interface PageLayoutProps {
  children: React.ReactNode;
  /** Subtitle shown in header */
  headerSubtitle?: string;
  /** Whether to show navigation in header */
  showNav?: boolean;
  /** Additional content for header right side */
  headerRightContent?: React.ReactNode;
  /** Footer tagline */
  footerTagline?: string;
}

export function PageLayout({
  children,
  headerSubtitle,
  showNav = true,
  headerRightContent,
  footerTagline,
}: PageLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#87CEEB] via-[#FFE4B5] to-[#98FB98] dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950 flex flex-col">
      <Header subtitle={headerSubtitle} showNav={showNav} rightContent={headerRightContent} />
      <main className="flex-1">{children}</main>
      <Footer tagline={footerTagline} />
    </div>
  );
}
