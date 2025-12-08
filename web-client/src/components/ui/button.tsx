import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground dark:bg-input/30 dark:border-input dark:hover:bg-input/50",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost:
          "hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",
        // 2005 Retro Gaming Variants - using font-bold and text-shadow for readability
        retro:
          "font-display font-bold bg-gradient-to-b from-[#FFE135] via-[#FFB300] to-[#FF8F00] text-[#5D4037] border-3 border-[#8B4513] rounded-xl shadow-[0_4px_0_#8B4513,0_6px_8px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] hover:translate-y-[-2px] hover:shadow-[0_6px_0_#8B4513,0_8px_12px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] active:translate-y-[2px] active:shadow-[0_2px_0_#8B4513,0_3px_4px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.2)] uppercase tracking-wide [text-shadow:1px_1px_0_rgba(255,255,255,0.5)]",
        "retro-green":
          "font-display font-bold bg-gradient-to-b from-[#7FFF00] via-[#32CD32] to-[#228B22] text-white border-3 border-[#006400] rounded-xl shadow-[0_4px_0_#006400,0_6px_8px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] hover:translate-y-[-2px] hover:shadow-[0_6px_0_#006400,0_8px_12px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] active:translate-y-[2px] active:shadow-[0_2px_0_#006400,0_3px_4px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.2)] uppercase tracking-wide [text-shadow:1px_1px_2px_rgba(0,0,0,0.4)]",
        "retro-pink":
          "font-display font-bold bg-gradient-to-b from-[#FF69B4] via-[#FF1493] to-[#C71585] text-white border-3 border-[#8B008B] rounded-xl shadow-[0_4px_0_#8B008B,0_6px_8px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] hover:translate-y-[-2px] hover:shadow-[0_6px_0_#8B008B,0_8px_12px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] active:translate-y-[2px] active:shadow-[0_2px_0_#8B008B,0_3px_4px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.2)] uppercase tracking-wide [text-shadow:1px_1px_2px_rgba(0,0,0,0.4)]",
        "retro-blue":
          "font-display font-bold bg-gradient-to-b from-[#00BFFF] via-[#1E90FF] to-[#0066CC] text-white border-3 border-[#00008B] rounded-xl shadow-[0_4px_0_#00008B,0_6px_8px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] hover:translate-y-[-2px] hover:shadow-[0_6px_0_#00008B,0_8px_12px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.4)] active:translate-y-[2px] active:shadow-[0_2px_0_#00008B,0_3px_4px_rgba(0,0,0,0.3),inset_0_2px_0_rgba(255,255,255,0.2)] uppercase tracking-wide [text-shadow:1px_1px_2px_rgba(0,0,0,0.4)]",
        "retro-outline":
          "font-display font-bold bg-white text-[#5D4037] border-3 border-[#8B4513] rounded-xl shadow-[0_4px_0_#8B4513,0_6px_8px_rgba(0,0,0,0.2)] hover:translate-y-[-2px] hover:shadow-[0_6px_0_#8B4513,0_8px_12px_rgba(0,0,0,0.2)] hover:bg-[#FFF9C4] active:translate-y-[2px] active:shadow-[0_2px_0_#8B4513,0_3px_4px_rgba(0,0,0,0.2)] uppercase tracking-wide [text-shadow:1px_1px_0_rgba(255,255,255,0.5)]",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        sm: "h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-md px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
        // Retro sizes (chunkier with larger, more readable fonts)
        "retro-sm": "h-10 rounded-xl gap-2 px-5 text-sm",
        "retro-default": "h-12 rounded-xl gap-2 px-6 text-base",
        "retro-lg": "h-14 rounded-xl gap-3 px-8 text-lg",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
