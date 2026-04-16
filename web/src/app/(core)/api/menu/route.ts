import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';

const INSTALL_APPS = (process.env.NEXTAPI_INSTALL_APP || 'example').split(',').map(app => app.trim());

interface MenuPatch {
  // dot-separated path of menu item "name" fields, e.g. "settings.security"
  target: string;
  children: any[];
}

// Recursively find a node by dot-separated name path and inject children into it
const applyPatch = (items: any[], targetParts: string[], children: any[]): boolean => {
  const [head, ...rest] = targetParts;
  for (const item of items) {
    if (item.name === head) {
      if (rest.length === 0) {
        item.children = [...(item.children || []), ...children];
        return true;
      }
      if (item.children) {
        if (applyPatch(item.children, rest, children)) return true;
      }
    }
  }
  return false;
};

const applyPatches = (menuItems: any[], patches: MenuPatch[]): void => {
  for (const patch of patches) {
    const targetParts = patch.target.split('.');
    const applied = applyPatch(menuItems, targetParts, patch.children);
    if (!applied) {
      console.warn(`[menu] patch target not found: ${patch.target}`);
    }
  }
};

const getDynamicMenuItems = async (locale: string) => {
  const dirPath = path.join(process.cwd(), 'src', 'app');

  let allMenuItems: any[] = [];
  const allPatches: MenuPatch[] = [];

  for (const app of INSTALL_APPS) {
    const menuPath = path.join(dirPath, app, 'constants', 'menu.json');

    try {
      await fs.access(menuPath);
      const menuContent = await fs.readFile(menuPath, 'utf-8');
      const menu = JSON.parse(menuContent);
      allMenuItems = allMenuItems.concat(menu[locale] || []);
      // Collect patches: key is "<locale>_patches"
      const patches: MenuPatch[] = menu[`${locale}_patches`] || [];
      allPatches.push(...patches);
    } catch {
      // Silently skip if menu.json doesn't exist for this app
    }
  }

  if (allPatches.length > 0) {
    applyPatches(allMenuItems, allPatches);
  }

  return allMenuItems;
};

const getFallbackMenuItems = async (locale: string) => {
  try {
    const localePath = path.join(process.cwd(), 'public', 'menus', `${locale}.json`);
    await fs.access(localePath);
    const menuContent = await fs.readFile(localePath, 'utf-8');
    const menus = JSON.parse(menuContent);
    return menus || [];
  } catch (err) {
    console.error(`Failed to load fallback menus for locale ${locale}:`, err);
    return [];
  }
};

export const GET = async (request: NextRequest) => {
  try {
    const { searchParams } = new URL(request.url);
    const locale = searchParams.get('locale') === 'en' ? 'en' : 'zh';

    let menuItems;

    try {
      menuItems = await getDynamicMenuItems(locale);
    } catch (error) {
      console.error('Error merging dynamic messages:', error);
    }

    if (!menuItems || menuItems.length === 0) {
      console.warn(`Fallback to public/menus for locale: ${locale}`);
      menuItems = await getFallbackMenuItems(locale);
    }

    return NextResponse.json(menuItems, { status: 200 });
  } catch (error) {
    console.error('Failed to load menus:', error);
    return NextResponse.json({ message: 'Failed to load menus', error }, { status: 500 });
  }
};

export const POST = async () => {
  return NextResponse.json({ message: 'Method Not Allowed' }, { status: 405 });
};
