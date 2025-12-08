import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Citrus, Github, Trophy, Gamepad2, Menu, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  /** Subtitle shown under the logo */
  subtitle?: string;
  /** Whether to show navigation buttons */
  showNav?: boolean;
  /** Additional content to render on the right side */
  rightContent?: React.ReactNode;
}

export function Header({ subtitle = 'LLM Decision-Making Benchmark', showNav = true, rightContent }: HeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="border-b-4 border-[#8B4513] bg-gradient-to-r from-[#FFE135] via-[#FFD700] to-[#FFA500] sticky top-0 z-50 shadow-[0_4px_0_#5D4037]">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 sm:gap-3 hover:scale-105 transition-transform">
            <div className="bg-white p-1.5 sm:p-2 rounded-xl border-3 border-[#8B4513] shadow-[3px_3px_0_#5D4037]">
              <Citrus className="h-6 w-6 sm:h-7 sm:w-7 text-[#FF6B35]" />
            </div>
            <div>
              <h1 className="font-display text-xl sm:text-2xl text-[#5D4037] drop-shadow-[2px_2px_0_rgba(255,255,255,0.5)]">
                LemonadeBench
              </h1>
              <p className="text-[10px] sm:text-xs font-semibold text-[#8B4513]/80 hidden sm:block">{subtitle}</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden sm:flex items-center gap-3">
            {showNav && (
              <>
                <a
                  href="https://github.com/Shaun3141/LemonadeBench"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#5D4037] hover:text-[#8B4513] transition-colors hover:scale-110"
                >
                  <Github className="h-6 w-6" />
                </a>
                <Link to="/leaderboard">
                  <Button variant="retro-outline" size="retro-sm" className="gap-2">
                    <Trophy className="h-4 w-4" />
                    Scores
                  </Button>
                </Link>
                <Link to="/game">
                  <Button variant="retro-green" size="retro-sm" className="gap-2">
                    <Gamepad2 className="h-4 w-4" />
                    Play!
                  </Button>
                </Link>
              </>
            )}
            {rightContent}
          </div>

          {/* Mobile Menu Button */}
          {showNav && (
            <button
              className="sm:hidden p-2 text-[#5D4037] hover:text-[#8B4513] transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          )}
          {/* Mobile: show rightContent if no nav */}
          {!showNav && rightContent && <div className="sm:hidden">{rightContent}</div>}
        </div>

        {/* Mobile Navigation Menu */}
        {showNav && mobileMenuOpen && (
          <div className="sm:hidden mt-3 pt-3 border-t-2 border-[#8B4513]/30">
            <div className="flex flex-col gap-2">
              <Link to="/game" onClick={() => setMobileMenuOpen(false)}>
                <Button variant="retro-green" size="retro-sm" className="w-full gap-2 justify-center">
                  <Gamepad2 className="h-4 w-4" />
                  Play!
                </Button>
              </Link>
              <Link to="/leaderboard" onClick={() => setMobileMenuOpen(false)}>
                <Button variant="retro-outline" size="retro-sm" className="w-full gap-2 justify-center">
                  <Trophy className="h-4 w-4" />
                  Scores
                </Button>
              </Link>
              <a
                href="https://github.com/Shaun3141/LemonadeBench"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 py-2 text-[#5D4037] hover:text-[#8B4513] font-semibold transition-colors"
              >
                <Github className="h-5 w-5" />
                GitHub
              </a>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
