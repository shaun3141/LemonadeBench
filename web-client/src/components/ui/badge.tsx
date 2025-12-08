import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center justify-center rounded-full border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,box-shadow] overflow-hidden",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground [a&]:hover:bg-primary/90",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground [a&]:hover:bg-secondary/90",
        destructive:
          "border-transparent bg-destructive text-white [a&]:hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground",
        // 2005 Retro Gaming Variants
        retro:
          "font-display border-2 border-[#8B4513] bg-gradient-to-b from-[#FFE135] to-[#FFB300] text-[#5D4037] shadow-[2px_2px_0_rgba(0,0,0,0.2)] uppercase tracking-wide px-3 py-1",
        "retro-gold":
          "font-display border-2 border-[#8B4513] bg-gradient-to-b from-[#FFD700] via-[#FFA500] to-[#FF8C00] text-[#5D4037] shadow-[2px_2px_0_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.5)] uppercase tracking-wide px-3 py-1",
        "retro-silver":
          "font-display border-2 border-[#696969] bg-gradient-to-b from-[#E8E8E8] via-[#C0C0C0] to-[#A8A8A8] text-[#333] shadow-[2px_2px_0_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.5)] uppercase tracking-wide px-3 py-1",
        "retro-bronze":
          "font-display border-2 border-[#5D3A1A] bg-gradient-to-b from-[#CD7F32] via-[#B87333] to-[#8B4513] text-white shadow-[2px_2px_0_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.3)] uppercase tracking-wide px-3 py-1",
        "retro-pink":
          "font-display border-2 border-[#8B008B] bg-gradient-to-b from-[#FF69B4] to-[#FF1493] text-white shadow-[2px_2px_0_rgba(0,0,0,0.2)] uppercase tracking-wide px-3 py-1",
        "retro-green":
          "font-display border-2 border-[#006400] bg-gradient-to-b from-[#7FFF00] to-[#32CD32] text-[#006400] shadow-[2px_2px_0_rgba(0,0,0,0.2)] uppercase tracking-wide px-3 py-1",
        "retro-blue":
          "font-display border-2 border-[#00008B] bg-gradient-to-b from-[#00BFFF] to-[#1E90FF] text-white shadow-[2px_2px_0_rgba(0,0,0,0.2)] uppercase tracking-wide px-3 py-1",
        "retro-pixel":
          "font-pixel text-[10px] border-2 border-[#333] bg-[#333] text-[#7FFF00] shadow-[2px_2px_0_#000] px-3 py-1",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : "span"

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
