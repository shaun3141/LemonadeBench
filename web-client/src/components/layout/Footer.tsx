import { Link } from 'react-router-dom';
import { Citrus } from 'lucide-react';

interface FooterProps {
  /** Additional text to show */
  tagline?: string;
}

export function Footer({ tagline = 'LLM Decision-Making Benchmark' }: FooterProps) {
  return (
    <footer className="border-t-4 border-[#8B4513] mt-auto py-3 sm:py-4 bg-gradient-to-r from-[#FFF9C4] via-[#FFECB3] to-[#FFF9C4] dark:bg-zinc-900/50">
      <div className="container mx-auto px-4 text-center text-xs sm:text-sm">
        <p className="mb-2 font-display text-[#5D4037] text-sm sm:text-base flex items-center justify-center gap-1.5">
          <Citrus className="h-4 w-4 sm:h-5 sm:w-5 text-[#FF6B35]" />
          LemonadeBench
          <span className="hidden sm:inline"> • {tagline}</span>
        </p>
        <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 sm:gap-4 text-[#8B4513]">
          <a
            href="https://github.com/Shaun3141/LemonadeBench"
            className="hover:text-[#FF6B35] font-semibold transition-colors"
          >
            GitHub
          </a>
          <span className="text-[#FFB300] hidden sm:inline">★</span>
          <a
            href="https://github.com/meta-pytorch/OpenEnv"
            className="hover:text-[#FF6B35] font-semibold transition-colors"
          >
            OpenEnv
          </a>
          <span className="text-[#FFB300] hidden sm:inline">★</span>
          <Link to="/leaderboard" className="hover:text-[#FF6B35] font-semibold transition-colors">
            Scores
          </Link>
          <span className="text-[#FFB300] hidden sm:inline">★</span>
          <Link to="/game" className="hover:text-[#FF6B35] font-semibold transition-colors">
            Play!
          </Link>
        </div>
      </div>
    </footer>
  );
}
