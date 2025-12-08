import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { BookOpen, Citrus, Sparkles, Snowflake, CupSoda, Timer, Check, Lightbulb, ShoppingCart, ThermometerSun, Refrigerator } from 'lucide-react';
import { CUPS_PER_LEMON, CUPS_PER_SUGAR_BAG, CUPS_PER_ICE_BAG } from '@/lib/constants';

interface RecipeDialogProps {
  disabled?: boolean;
}

export function RecipeDialog({ disabled }: RecipeDialogProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 gap-1.5" disabled={disabled}>
          <BookOpen className="h-3.5 w-3.5" />
          Recipe
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-purple-600" />
            Lemonade Recipe
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <p className="text-sm text-muted-foreground">
            Each cup of lemonade requires these ingredients:
          </p>
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-950/30 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <Citrus className="h-8 w-8 mx-auto text-yellow-500 mb-2" />
              <div className="text-2xl font-bold">1</div>
              <div className="text-sm font-medium">Lemon</div>
              <div className="text-xs text-muted-foreground mt-1">
                makes {CUPS_PER_LEMON} cups
              </div>
              <div className="text-xs text-yellow-600 mt-1">$0.25 each</div>
              <div className="text-xs text-red-500 mt-1 flex items-center justify-center gap-1">
                <Timer className="h-3 w-3" /> Expires in 3 days
              </div>
            </div>
            <div className="text-center p-4 bg-pink-50 dark:bg-pink-950/30 rounded-lg border border-pink-200 dark:border-pink-800">
              <Sparkles className="h-8 w-8 mx-auto text-pink-400 mb-2" />
              <div className="text-2xl font-bold">1</div>
              <div className="text-sm font-medium">Sugar Bag</div>
              <div className="text-xs text-muted-foreground mt-1">
                makes {CUPS_PER_SUGAR_BAG} cups
              </div>
              <div className="text-xs text-pink-600 mt-1">$1.00 each</div>
              <div className="text-xs text-green-600 mt-1 flex items-center justify-center gap-1">
                <Check className="h-3 w-3" /> Never expires
              </div>
            </div>
            <div className="text-center p-4 bg-cyan-50 dark:bg-cyan-950/30 rounded-lg border border-cyan-200 dark:border-cyan-800">
              <Snowflake className="h-8 w-8 mx-auto text-cyan-400 mb-2" />
              <div className="text-2xl font-bold">1</div>
              <div className="text-sm font-medium">Ice Bag</div>
              <div className="text-xs text-muted-foreground mt-1">
                makes {CUPS_PER_ICE_BAG} cups
              </div>
              <div className="text-xs text-cyan-600 mt-1">$0.50 each</div>
              <div className="text-xs text-red-500 mt-1 flex items-center justify-center gap-1">
                <Timer className="h-3 w-3" /> Melts overnight!
              </div>
            </div>
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
              <CupSoda className="h-8 w-8 mx-auto text-blue-400 mb-2" />
              <div className="text-2xl font-bold">1</div>
              <div className="text-sm font-medium">Cup</div>
              <div className="text-xs text-muted-foreground mt-1">= 1 serving</div>
              <div className="text-xs text-blue-600 mt-1">$0.05 each</div>
              <div className="text-xs text-green-600 mt-1 flex items-center justify-center gap-1">
                <Check className="h-3 w-3" /> Never expires
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg p-3 text-sm">
            <div className="font-medium mb-2 flex items-center gap-1.5">
              <Lightbulb className="h-4 w-4 text-yellow-500" />
              Tips:
            </div>
            <ul className="text-xs text-muted-foreground space-y-1.5">
              <li className="flex items-start gap-1.5">
                <ShoppingCart className="h-3 w-3 mt-0.5 flex-shrink-0 text-green-500" />
                <span><strong>Buy in bulk</strong> to save up to 20%!</span>
              </li>
              <li className="flex items-start gap-1.5">
                <ThermometerSun className="h-3 w-3 mt-0.5 flex-shrink-0 text-orange-500" />
                <span><strong>Ice boosts demand</strong> on hot/sunny days (+20% customers!)</span>
              </li>
              <li className="flex items-start gap-1.5">
                <Refrigerator className="h-3 w-3 mt-0.5 flex-shrink-0 text-cyan-500" />
                <span>Ice melts overnight - buy a <strong>cooler</strong> to preserve 50%!</span>
              </li>
              <li className="flex items-start gap-1.5">
                <Timer className="h-3 w-3 mt-0.5 flex-shrink-0 text-yellow-500" />
                <span>Lemons last 3 days - plan purchases carefully</span>
              </li>
              <li className="flex items-start gap-1.5">
                <Check className="h-3 w-3 mt-0.5 flex-shrink-0 text-green-500" />
                <span>Sugar and cups never expire - stock up!</span>
              </li>
            </ul>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
