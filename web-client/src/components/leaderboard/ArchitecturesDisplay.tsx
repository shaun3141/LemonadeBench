import { Card, CardContent } from '@/components/ui/card';
import { ARCHITECTURES } from './constants';

export function ArchitecturesDisplay() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {ARCHITECTURES.map((arch) => {
        const Icon = arch.icon;
        return (
          <Card key={arch.id} variant="retro-blue" className="overflow-hidden">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="p-2 rounded-xl bg-[#1976D2]/20 border-2 border-[#1976D2]">
                  <Icon className="h-4 w-4 text-[#1976D2]" />
                </div>
                <h4 className="font-display text-[#0D47A1]">{arch.name}</h4>
              </div>
              <p className="text-xs font-pixel bg-[#0D47A1] text-[#7FFF00] px-2 py-1 rounded mb-2 shadow-[2px_2px_0_#000]">
                {arch.flow}
              </p>
              <p className="text-sm text-[#0D47A1]/70">{arch.description}</p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

