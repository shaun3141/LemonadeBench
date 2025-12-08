import { Card, CardContent } from '@/components/ui/card';
import { SCAFFOLDINGS } from './constants';

export function ScaffoldingsDisplay() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {SCAFFOLDINGS.map((scaff) => {
        const Icon = scaff.icon;
        return (
          <Card key={scaff.id} variant="retro-green" className="overflow-hidden">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="p-2 rounded-xl bg-[#388E3C]/20 border-2 border-[#388E3C]">
                  <Icon className="h-4 w-4 text-[#388E3C]" />
                </div>
                <h4 className="font-display text-[#1B5E20]">{scaff.name}</h4>
              </div>
              <p className="text-sm text-[#1B5E20]/70">{scaff.description}</p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

