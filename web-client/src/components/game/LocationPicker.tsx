import { Badge } from '@/components/ui/badge';
import type { LocationId, LocationInfo } from '@/types';
import { getLocationIcon, getLocationBlurb, LOCATION_STAT_ICONS } from '@/lib/locations';
import { formatCents } from '@/lib/format';
import { MapPinOff } from 'lucide-react';

interface LocationPickerProps {
  locations: LocationInfo[];
  currentLocation: LocationId;
  selectedLocation: LocationId | null;
  onSelectLocation: (locationId: LocationId | null) => void;
  disabled?: boolean;
  isGameOver?: boolean;
}

export function LocationPicker({
  locations,
  currentLocation,
  selectedLocation,
  onSelectLocation,
  disabled,
  isGameOver,
}: LocationPickerProps) {
  if (!locations || locations.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <MapPinOff className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No locations available yet</p>
      </div>
    );
  }

  const TrafficIcon = LOCATION_STAT_ICONS.traffic;
  const WeatherIcon = LOCATION_STAT_ICONS.weather;
  const PriceIcon = LOCATION_STAT_ICONS.price;
  const PermitIcon = LOCATION_STAT_ICONS.permit;

  const isChangingLocation = selectedLocation !== null && selectedLocation !== currentLocation;
  const locationCost = isChangingLocation
    ? locations.find((l) => l.id === selectedLocation)?.permit_cost ?? 0
    : 0;

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        {locations.map((location) => {
          const isCurrent = location.is_current;
          const isSelected = selectedLocation === location.id;
          const willBeLocation = isSelected || (selectedLocation === null && isCurrent);

          const LocationIcon = getLocationIcon(location.id);
          const blurb = getLocationBlurb(location);

          return (
            <div
              key={location.id}
              onClick={() => {
                if (!disabled && !isGameOver) {
                  if (isCurrent && selectedLocation === null) {
                    // Already at this location and no change selected - do nothing
                  } else if (isSelected) {
                    // Deselect (stay at current)
                    onSelectLocation(null);
                  } else {
                    // Select this location
                    onSelectLocation(location.id);
                  }
                }
              }}
              className={`p-3 rounded-lg border-2 transition-all cursor-pointer ${
                willBeLocation
                  ? 'bg-emerald-100 dark:bg-emerald-950/30 border-emerald-400 dark:border-emerald-600 ring-2 ring-emerald-400'
                  : 'bg-gray-100 dark:bg-gray-800/50 border-gray-300 dark:border-gray-600 hover:border-emerald-400'
              } ${disabled || isGameOver ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-start gap-2">
                <LocationIcon
                  className={`h-5 w-5 flex-shrink-0 mt-0.5 ${willBeLocation ? 'text-emerald-600' : 'text-gray-500'}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-sm">{location.name}</span>
                    {isCurrent && (
                      <Badge className="bg-emerald-500 text-white text-[10px] px-1.5 py-0 h-4">
                        Current
                      </Badge>
                    )}
                  </div>

                  {/* Location Details */}
                  <div className="mt-1.5 space-y-0.5 text-[11px] text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <TrafficIcon className="h-3 w-3 text-blue-500" />
                      {blurb.trafficDesc}
                    </div>
                    <div className="flex items-center gap-1">
                      <WeatherIcon className="h-3 w-3 text-orange-500" />
                      {blurb.weatherDesc}
                    </div>
                    <div className="flex items-center gap-1">
                      <PriceIcon className="h-3 w-3 text-green-500" />
                      {blurb.priceDesc}
                    </div>
                    {!isCurrent && (
                      <div className="flex items-center gap-1">
                        <PermitIcon className="h-3 w-3 text-purple-500" />
                        {blurb.costDesc}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {isChangingLocation && (
        <div className="text-xs text-orange-600 dark:text-orange-400 text-center pt-1">
          Moving requires a {formatCents(locationCost)} permit fee
        </div>
      )}
    </div>
  );
}
