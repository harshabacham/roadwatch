import { Camera, FileText, Globe, MapPin } from 'lucide-react';
import { PWAInstallButton } from './PWAInstallButton';

export type ActiveTab = 'scanner' | 'map' | 'complaints' | 'portals';

interface TopNavProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  complaintCount: number;
}

export function TopNav({
  activeTab,
  onSelectTab,
  complaintCount,
}: TopNavProps) {
  const navItems: { id: ActiveTab; label: string }[] = [
    { id: 'scanner', label: 'Live Scanner' },
    { id: 'map', label: 'Hazard Map' },
    {
      id: 'complaints',
      label: complaintCount > 0 ? `Complaints (${complaintCount})` : 'Complaints',
    },
    { id: 'portals', label: 'Govt Portals' },
  ];

  const mobileTabs: { id: ActiveTab; label: string; icon: typeof Camera }[] = [
    { id: 'scanner', label: 'Scanner', icon: Camera },
    { id: 'map', label: 'Map', icon: MapPin },
    { id: 'complaints', label: 'Reports', icon: FileText },
    { id: 'portals', label: 'Portals', icon: Globe },
  ];

  return (
    <>
      <header className="no-print sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-13 md:h-16 flex items-center justify-between gap-4 sm:gap-8">
          {/* Zone 1: Single-element brand wordmark */}
          <button
            type="button"
            onClick={() => onSelectTab('scanner')}
            className="font-display text-lg md:text-xl font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-600 rounded"
          >
            RoadWatch
          </button>

          {/* Zone 2: Clean text navigation links (desktop) */}
          <nav
            aria-label="Primary navigation"
            className="hidden md:flex items-center gap-6 text-sm font-medium"
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectTab(item.id)}
                  className={`py-1.5 border-b-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-600 ${
                    isActive
                      ? 'border-amber-600 text-slate-900 font-semibold'
                      : 'border-transparent text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: Install & primary action */}
          <div className="flex items-center gap-2 shrink-0">
            <PWAInstallButton />
            <button
              type="button"
              onClick={() =>
                onSelectTab(activeTab === 'scanner' ? 'complaints' : 'scanner')
              }
              className="px-4 py-2 text-xs sm:text-sm font-semibold text-white bg-amber-600 hover:bg-amber-700 active:scale-[0.98] rounded-lg transition-all whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-600 focus-visible:ring-offset-2"
            >
              {activeTab === 'scanner' ? 'Dispatch Queue' : 'Open Scanner'}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile bottom thumb-zone tab bar */}
      <nav
        aria-label="Mobile navigation"
        className="no-print md:hidden fixed bottom-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md border-t border-slate-200 pb-safe"
      >
        <div className="grid grid-cols-4 h-15 items-center">
          {mobileTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onSelectTab(tab.id)}
                className={`min-h-[44px] flex flex-col items-center justify-center transition-colors cursor-pointer ${
                  isActive ? 'text-amber-600' : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                <span className="relative">
                  <Icon className="w-5 h-5" strokeWidth={isActive ? 2.25 : 1.75} />
                  {tab.id === 'complaints' && complaintCount > 0 && (
                    <span className="absolute -top-1.5 -right-2.5 px-1 min-w-[16px] h-4 rounded-full bg-amber-600 text-white text-[10px] font-mono font-semibold flex items-center justify-center">
                      {complaintCount > 99 ? '99+' : complaintCount}
                    </span>
                  )}
                </span>
                <span className="text-[11px] font-medium tracking-tight mt-0.5 whitespace-nowrap">
                  {tab.label}
                </span>
              </button>
            );
          })}
        </div>
      </nav>
    </>
  );
}
